# Agent Memory System

A persistent memory layer for a conversational LLM agent — built to make responses feel
personalized and consistent across sessions, without adding noticeable latency and without
ever confidently telling the model something false or stale about the user.

This is a from-scratch rebuild. The previous implementation is gone; this document is the
plan the rebuild follows. Nothing below is implemented yet — this is the spec.

This project is built and distributed as an **installable Python library published on PyPI**
(`pip install agent-memory`, name TBD — see Open Decisions), not as a standalone service. A
host application (FastAPI, Flask, a CLI, whatever) imports it and calls it directly. This
constraint shapes several decisions below: storage and LLM backends must be pluggable rather
than hardcoded, the formation worker must be something the host process runs rather than
something the library owns, and the public API surface has to be small and stable since other
people's code will depend on it.

Design synthesized from production/research systems: **Mem0** (extraction + ADD/UPDATE/DELETE/NOOP
pipeline), **Zep/Graphiti** (bi-temporal knowledge graph), **Letta/MemGPT** (tiered,
OS-inspired memory), and Stanford's **Generative Agents** (reflection / memory synthesis).

---

## 1. Problem statement

Given `(user_id, new_message)`, produce a response that reflects everything worth knowing
about this user from past interactions — without the user waiting for that "remembering" to
happen, and without the system ever holding two conflicting "truths" about the user at once.

Two things matter equally: **speed** (the user is waiting) and **accuracy** (a wrong or stale
memory actively makes the agent worse, not neutral).

## 2. Core design principle

**Reading memory and forming memory are different problems with different cost budgets, and
must never share a code path.**

- **Read path** — runs between "user hits send" and "model starts responding." Hard latency
  budget. No LLM reasoning about *what* to retrieve — only cache reads, index lookups, and
  arithmetic scoring.
- **Formation path** — runs after the response has already been sent. No latency budget. This
  is where all the expensive reasoning (contradiction resolution, confidence judgment,
  deduplication) is allowed to happen, because nobody is waiting on it.

Two independent services connected by a durable queue, not one pipeline with async bits
bolted on.

## 3. Memory tiers

| Tier | Contents | Storage | Access pattern |
|---|---|---|---|
| **Tier 0 — Session** | Rolling recent turns, active task state | Redis | O(1) read, per-conversation key |
| **Tier 1 — Core profile** | Small, precomputed, always-injected user profile | Redis (backed by Postgres) | O(1) read, whole blob, no search |
| **Tier 2 — Active store** | Extracted facts (vector) + entities/relationships (bi-temporal graph) + keyword index | Postgres (pgvector + edges table) | Parallel vector / graph / keyword query |
| **Tier 3 — Reflections** | Higher-level patterns synthesized from clusters of Tier 2 facts | Postgres | Retrieved like any other memory |
| **Archival** | Decayed-out, low-relevance, or old memory | Postgres (cold) | Never on the hot path; audit/debug only |

Tier 2 is deliberately one logical store with two representations of the same facts, not two
separate subsystems:
- **Vector** (Mem0-style): flat facts + embeddings, for fuzzy semantic recall ("what did we
  discuss about pricing").
- **Graph** (Zep-style): entities + relationships as **bi-temporal edges** —
  `(source, relation, target, valid_from, valid_to, observed_at, recorded_at)`. Contradictions
  never delete a row: a new fact closes the old edge's `valid_to` and inserts a new edge.
  "Current truth" is just `valid_to IS NULL`. Full history is preserved for free.
- **Keyword/BM25** over the same store, run in parallel with the other two — catches exact
  names/IDs that embeddings sometimes miss.

## 4. Request-time workflow (read path)

Everything here is cache, index, or arithmetic — never an LLM call deciding *what* to fetch.

1. **Retrieval gate** (heuristic, not LLM): skip the entire retrieval pipeline for turns that
   obviously don't need memory ("ok", "continue", "thanks").
2. **Parallel fetch**: Tier 0 + Tier 1 reads, plus one query embedding computed once and reused
   across all three Tier 2 channels (vector, graph, keyword) — fired concurrently, never in a
   sequential loop.
3. **Two-stage funnel**: fast approximate fetch (ANN top-20 via HNSW) → deterministic rerank:
   `score = w1·relevance + w2·recency_decay + w3·importance + w4·type_weight`.
4. **Pack**: top-N by score, greedily filled into a hard token budget (not a fixed item count).
5. **Assemble**: deterministic prompt template — Tier 0 + Tier 1 + packed Tier 2/3 results + new
   message.
6. **One LLM call** → response.
7. Return response to the user. Push `{user_id, conversation_id, turn}` onto the durable queue
   — fire-and-forget, does not block the response.

Floor cost: 2 cache reads (parallel) + 1 embedding + 3 parallel index lookups + 1 rerank pass +
1 LLM call. This is the actual speed ceiling — not an implementation detail to optimize later.

## 5. Formation workflow (write path, async)

Consumes turn-completed events from the durable queue, one turn at a time, per user.

1. **Extract**: one structured-output LLM call → typed candidates (fact / relation /
   preference), each with a confidence score and an explicit-vs-inferred flag.
2. **Resolve**: for each candidate, retrieve its top-k nearest existing memories (same
   retrieval mechanism as the read path, reused).
3. **Classify operation**: one LLM tool-call decides `ADD / UPDATE / DELETE / NOOP` against
   those candidates (Mem0's mechanism).
4. **Safety gate** (deterministic, not LLM): identity- and constraint-class fields must
   additionally pass an explicit-statement-or-N-repetitions check regardless of step 3's
   decision. Prevents one bad extraction from silently overwriting who the user is.
5. **Write**:
   - Relational fact → bi-temporal edge (see Tier 2 above).
   - Flat fact → vector row, with confidence + observation_count. Merge into an existing row
     above 0.85 cosine similarity instead of inserting a duplicate.
6. **Reflection** (batched — e.g. hourly per active user, never per-turn): cluster recent
   writes, synthesize a Tier 3 summary where a pattern has emerged across ≥N observations.
7. **Decay sweep** (batched — e.g. daily): old, unreinforced, unretrieved Tier 2 rows move to
   Archival. This keeps the active HNSW index small, which is what keeps step 3 of the read
   path fast as the system ages — decay and speed are the same mechanism.

## 6. Storage

- **Postgres**: `turns` (raw, append-only, source of truth) · `memory_facts` (Tier 2 vector
  rows) · `memory_edges` (Tier 2 bi-temporal graph) · `reflections` (Tier 3) · `archival_*`
  (cold copies). pgvector + HNSW index for vector search. Plain indexed edges table with
  recursive CTEs for 1–2 hop graph queries — no separate graph database at this scale.
- **Redis**: Tier 0 session cache, Tier 1 profile cache, durable job stream (Redis Streams)
  feeding the formation worker pool.
- **Formation workers**: a separate deployable from the API, scaled independently, so a
  restart never silently drops queued learning work.

## 7. Non-negotiables

- **User-facing visibility/control**: view, edit, delete stored memories. Both ChatGPT and
  Claude treat this as core product surface, not an afterthought — it also doubles as the
  primary debugging tool during development.
- **Observability**: structured logs (not `print`) and a full memory-operation audit trail —
  every ADD/UPDATE/DELETE/NOOP and every edge invalidation logged with its reasoning. This is
  the only way to see the system's judgment after the fact, since none of it is visible in the
  final response.
- **Per-user isolation**: all storage and queue partitioning keyed by `user_id`, so one user's
  write load never contends with another's reads.

## 8. Build order

Do not build all tiers at once. Per production precedent (Mem0/Zep's own staged rollouts):

**Phase 1 (MVP)**
- Tier 0 (session cache) + Tier 1 (core profile)
- Tier 2, vector half only (flat facts + embeddings, no graph yet)
- Formation pipeline: extract → resolve → ADD/UPDATE/DELETE/NOOP → safety gate
- Basic decay sweep
- User-facing memory view/edit/delete

This alone should deliver the large majority of the latency and accuracy win.

**Phase 2**
- Tier 2 graph half (bi-temporal edges) + keyword/BM25 channel
- Tier 3 reflections
- Full observability/audit trail

Graduate to Phase 2 only once real usage data from Phase 1 shows where flat-vector retrieval
is actually falling short — not speculatively upfront.

## 9. Packaging: distributed as a PyPI library

**Repo layout — `src` layout (standard for publishable packages, avoids accidentally testing
against the working directory instead of the installed package):**

```
agent-memory-system/                 (repo root)
├── pyproject.toml                   (PEP 621 metadata, build backend, optional-dependencies)
├── README.md
├── LICENSE
├── src/
│   └── agent_memory/                (importable package — the actual library)
│       ├── __init__.py              (small public API surface: AgentMemory, config types)
│       ├── py.typed                 (marks the package as type-hinted for downstream users)
│       ├── config.py                (settings/config objects, no global state)
│       ├── read/                    (read-path: Section 4)
│       │   ├── gate.py
│       │   ├── session_cache.py
│       │   ├── profile_cache.py
│       │   ├── retrieval.py
│       │   └── rerank.py
│       ├── formation/               (write-path: Section 5)
│       │   ├── extract.py
│       │   ├── resolve.py
│       │   ├── operations.py        (ADD/UPDATE/DELETE/NOOP)
│       │   ├── safety_gate.py
│       │   ├── reflection.py
│       │   ├── decay.py
│       │   └── worker.py            (exposes run_formation_worker() — host process runs this)
│       ├── storage/
│       │   ├── interfaces.py        (abstract backend protocols)
│       │   ├── postgres/            (facts, edges, reflections, migrations)
│       │   └── redis/               (session cache, profile cache, job stream)
│       ├── llm/
│       │   └── interfaces.py        (provider-agnostic LLM + embedding client protocols)
│       └── models/                  (shared pydantic schemas)
├── tests/
│   ├── unit/
│   └── integration/
└── examples/
    └── fastapi_app/                 (reference integration: how a host app wires this in)
```

**Packaging decisions this implies:**

- **Storage backends are pluggable via interfaces** (`storage/interfaces.py`), with Postgres +
  Redis shipped as the default implementations — a library consumer isn't forced onto our
  exact infra choices, though those remain the recommended default.
- **LLM/embedding providers are pluggable** the same way (`llm/interfaces.py`) — AWS Bedrock,
  OpenAI, Anthropic, or a local embedding model can all satisfy the same protocol. No hardcoded
  provider inside the core package.
- **The formation worker is exposed, not owned.** The library provides
  `run_formation_worker()`; the host application decides whether to run it as an in-process
  asyncio task (simple deployments) or as a separate process/service (Phase 1 build-order
  default per Section 8) — the library doesn't assume either.
- **Optional extras** in `pyproject.toml` so installing the library doesn't force every
  dependency: e.g. `pip install agent-memory[postgres,redis,bedrock]`.
- **Semantic versioning** from the first published release, since a public API surface means
  breaking changes have real downstream cost.
- `examples/fastapi_app` is a reference/demo of integrating the library into a service — it is
  not part of the published package.

## 10. Open decisions (to confirm before/while building)

- LLM provider and model for generation, extraction, and operation-classification calls.
- Embedding model: local (e.g. sentence-transformers, in-process) vs. hosted API — local
  avoids a network hop on the one embedding call that sits on the critical path.
- Deployment target for the formation worker pool (separate process vs. separate service).
- **PyPI package name** — `agent-memory` may already be taken; needs a availability check
  before it's final. Import name (`agent_memory` vs. something else) should match.
- **Minimum supported Python version** and how far back to support (affects typing syntax,
  async features available).
