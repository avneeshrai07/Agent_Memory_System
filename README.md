# memory-verse-avneesh

A persistent memory layer for a conversational LLM agent — built to make responses feel
personalized and consistent across sessions, without adding noticeable latency and without
ever confidently telling the model something false or stale about the user.

This is a from-scratch rebuild. The previous implementation is gone; this document is the
plan the rebuild follows.

This project is built and distributed as an **installable Python library published on PyPI**
(`pip install memory-verse-avneesh`), not as a standalone service — the third package in the
`-verse-avneesh` family, alongside `storage-verse-avneesh` and `llm-verse-avneesh`. A
host application (FastAPI, Flask, a CLI, whatever) imports it and calls it directly. This
constraint shapes several decisions below: storage and LLM backends must be pluggable rather
than hardcoded, the formation worker must be something the host process runs rather than
something the library owns, and the public API surface has to be small and stable since other
people's code will depend on it.

Design synthesized from production/research systems: **Mem0** (extraction + ADD/UPDATE/DELETE/NOOP
pipeline), **Zep/Graphiti** (bi-temporal knowledge graph), **Letta/MemGPT** (tiered,
OS-inspired memory), and Stanford's **Generative Agents** (reflection / memory synthesis).

---

## 0. Quick start

The recommended way to start: `connect()` does all the wiring in one call — creates the
Postgres pool, creates every table (facts, identity, episodic, reminders, graph) and ensures
their schema, builds the Tier 0/1 cache backend, and constructs every LLM client. It hands back
a single `Memory` object whose `.read()`/`.write()` already have every backend bound, so a
request only needs to pass what's actually request-specific:

```python
from memory_verse_avneesh import connect

memory = await connect(
    database_url=DATABASE_URL,      # or postgres_host/port/user/password/database
    postgres_schema="my_app",       # required, no default -- see Section 6
    upstash_url=UPSTASH_URL,        # or redis_url
    upstash_token=UPSTASH_TOKEN,
    aws_region="us-east-1",
    aws_llm_access_key_id=AWS_KEY,      # optional -- omit to use boto3's default credential chain
    aws_llm_secret_access_key=AWS_SECRET,
)

# request path
context = await memory.read(user_id=user_id, conversation_id=conversation_id, message=message)
prompt = memory.render_prompt(context, message)
response_text = ...  # your own generation call -- memory_verse_avneesh has no part in this

# after generation completes
turn = Turn(user_id=user_id, conversation_id=conversation_id,
            user_message=message, assistant_message=response_text)
await memory.session_cache.append_turn(turn)
background_tasks.add_task(memory.write, turn)  # backgrounded, never blocks the response

# on shutdown
await memory.close()
```

LLM calls go straight to AWS Bedrock, and the Tier 0/1 cache goes straight to Redis or
Upstash (whichever you configured) — no extra dependency beyond this library's own
`postgres`/`redis`\|`upstash`/`bedrock` extras.

Every underlying store/client is still a public attribute on `Memory` (`memory.fact_store`,
`memory.identity_store`, `memory.episodic_store`, `memory.reminder_store`, `memory.graph_store`,
`memory.session_cache`, `memory.profile_cache`, `memory.embedding_client`, ...) — reach for these
directly when calling `memory_verse_avneesh.identity`/`episodic`/`prospective`/`graph`/
`management`'s own functions, e.g. `create_expert_identity(..., identity_store=memory.identity_store)`.

Building `MemoryConfig` and each backend by hand (see `examples/fastapi_app`) is still fully
supported for hosts that need custom wiring `connect()` doesn't cover — a non-default pool
size, a swapped-in LLM client implementation, or credentials that don't fit `connect()`'s
flat argument list. Everything below this section documents that manual
path and the architecture behind both.

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

**A third boundary, specific to this being a library rather than a service**: the read path
itself stops at *retrieving* memory — it does not generate the user-facing response. The
library hands back structured memory context; the host application makes its own generation
call (its own model, tools, streaming, provider) and, once it has a response, builds the `Turn`
and hands it to the formation path itself. The library never calls an LLM to produce a
response a user sees — only to retrieve (embeddings) or to reason about what's true (formation
extraction/classification).

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
- **Graph** (Zep/Graphiti-style): entities + relationships as **bi-temporal edges** —
  `(source, relation, target, valid_from, valid_to, observed_at, recorded_at)`. Contradictions
  never delete a row: a new fact closes the old edge's `valid_to` and inserts a new edge.
  "Current truth" is just `valid_to IS NULL`. Full history is preserved for free. Edges are
  retrieved by embedding a deterministically-templated **fact sentence** per edge (e.g. "User
  works at Acme Corp"), not the bare entity names — matching Zep/Graphiti's precedent, since a
  conversational query matches a full relationship sentence far better than a short name string.
  Entity resolution is exact case-insensitive name/alias matching in this version (no
  fuzzy/embedding-based resolution yet). Edges are treated as single-valued per
  `(source_entity_id, relation)` — a new candidate for the same pair always supersedes the
  current one; multi-valued relations (e.g. "friends_with" allowing several concurrent targets)
  aren't modeled specially. One caveat worth naming: the relation string itself is chosen freely
  by the extraction LLM call, not drawn from a fixed vocabulary — if it phrases the same
  real-world relationship differently across distant turns (`"managed_by"` vs. `"has_manager"`),
  contradiction detection (which matches on the literal relation string) won't catch it, and
  parallel "current" edges can result instead of a clean supersession.
- **Keyword/BM25** over the same store, run in parallel with the other two — catches exact
  names/IDs that embeddings sometimes miss. Not yet built (see Section 8).

**Episodic memory** is also separate from the Tier 2 table above — a durable, embedded record
of every turn (`episodes` table), not just the distilled facts extracted from it. Written
unconditionally by `write_memory()` for every turn, with no LLM judgment about what's "worth
remembering" (that's what fact extraction already does; episodic memory's value is completeness
— an actual answer to "what happened, when," not just "what do I know about the user"). Searched
the same way as Tier 2 facts (embedding + ANN + rerank), sharing the same query embedding and
retrieval gate, but reranked by relevance + recency only (no confidence/type weighting — an
episode doesn't have those). Immutable except for explicit user-requested deletion; no decay,
since it's the audit trail Tier 2 facts get distilled *from*, not a duplicate of Tier 2 itself.

**Identity** is a separate concept from the tiers above, not a tier itself — two distinct
Postgres tables, neither written by the formation pipeline:
- **Expert identity**: host-authored personas ("expert_email_writer"), keyed by a string id the
  host chooses, full CRUD via `memory_verse_avneesh.identity`. Selected explicitly per
  `read_memory()` call via `identity_id` — never auto-selected.
- **Person identity**: one durable record per `user_id`, distinct from the Tier 1 profile cache
  (that's an ephemeral, formation-derived blob; this is a deliberate record the host writes),
  always fetched automatically by `read_memory()` when an `IdentityStore` is configured.

Both surface on `MemoryContext` as `expert_identity` / `person_identity` — combined together
when both are present, e.g. an "expert email writer" persona applied with a specific person's
own tone preferences layered on top.

**Prospective memory** (`reminders` table) is future intentions, not facts about the past or
present — "do X later." Created explicitly through `memory_verse_avneesh.prospective`, by the
host's own code or its own LLM calling it as a tool during generation; `write_memory()` never
creates one automatically, there is no "this sounds like something to remind them about"
extraction in this version. `read_memory()` always includes PENDING reminders with
`due_at <= now` on `MemoryContext.due_reminders` when a `ReminderStore` is configured — a plain
deterministic time comparison, not a similarity search, so it's included regardless of the
current message's content (even a trivial "thanks!" still surfaces a due reminder). A reminder
stays PENDING — and keeps being returned — until explicitly marked done or dismissed; passing
`due_at` doesn't silently remove it.

## 4. Request-time workflow (read path)

Steps 1–4 are the library's job (`read_memory()`) — cache, index, or arithmetic only,
never an LLM call deciding *what* to fetch. Steps 5–6 are the host application's own code,
built on what the library returns; the library does not do them.

1. **Retrieval gate** (heuristic, not LLM): skip Tier 2 (the embedding call + vector search)
   for turns that obviously don't need durable memory ("ok", "thanks") — Tier 0/1 are always
   read regardless, deliberately: they're O(1) cache reads, and dropping them on a one-word
   reply would break conversational continuity for no real speed win. Tier 2 is the part
   actually worth skipping.
2. **Parallel fetch**: Tier 0 + Tier 1 reads, plus (gate permitting) one query embedding
   computed once and reused across all three Tier 2 channels (vector, graph, keyword) — fired
   concurrently, never in a sequential loop.
3. **Two-stage funnel**: fast approximate fetch (ANN top-20 via HNSW) → deterministic rerank:
   `score = w1·relevance + w2·recency_decay + w3·importance + w4·type_weight`.
4. **Return structured context** (`MemoryContext`: profile + ranked facts + recent turns +
   ranked episodes + ranked relationship edges + identity + due reminders), packed to a token
   budget (facts, episodes, and edges each have independent budgets; due reminders aren't
   budget-packed, since they're a plain time filter, not a ranked/truncated list). Edge
   retrieval additionally does a bounded one-hop expansion: after matching edges by
   fact-sentence similarity, it also pulls in other current edges touching the same entities,
   so a direct match like "managed by David" can surface a connected fact like "works at Acme
   Corp" one hop away — reranked alongside the direct matches, not treated as equally relevant.
   An optional convenience can flatten this to text, but the structured form is the real
   contract — the library's responsibility ends here.

*— host-owned, outside the library —*

5. **Generate**: the host builds its own prompt/messages from the returned context (its own
   system prompt, tools, streaming, model, provider) and makes its own generation call.
6. **Persist + hand off**: once the host has its own response, it constructs the `Turn` (it has
   both messages now), calls `SessionCache.append_turn()`, and pushes the turn to formation —
   fire-and-forget, so it never blocks the response already returned to the user.

Floor cost through step 4 (the part the library is responsible for): 2 cache reads (parallel) +
1 embedding + 3 parallel index lookups + 1 rerank pass. This is the speed ceiling the library
controls; generation latency (step 5) is the host's own model choice, not the library's to own
or optimize.

## 5. Formation workflow (write path, async)

Consumes turn-completed events from the durable queue, one turn at a time, per user. Each
`Turn` was constructed by the host application (README Section 4, step 6) after its own
generation call — the library only ever sees a turn once both messages already exist.

**Flat facts** (Tier 2 vector):
1. **Extract**: one structured-output LLM call (`ExtractionClient`) → typed candidates, each
   with a confidence score and an explicit-vs-inferred flag.
2. **Resolve**: for each candidate, retrieve its top-k nearest existing facts (same retrieval
   mechanism as the read path, reused).
3. **Classify operation**: one LLM tool-call (`ResolutionClient`) decides
   `ADD / UPDATE / DELETE / NOOP` against those candidates (Mem0's mechanism).
4. **Safety gate** (deterministic, not LLM): identity- and constraint-class fields must
   additionally pass an explicit-statement-or-N-repetitions check regardless of step 3's
   decision. Prevents one bad extraction from silently overwriting who the user is.
5. **Write**: vector row, with confidence + observation_count. Merge into an existing row
   above 0.85 cosine similarity instead of inserting a duplicate.

**Relationships** (Tier 2 graph) — a separate pipeline, not a branch of the one above:
1. **Extract**: one structured-output LLM call (`RelationExtractionClient`) → `(source,
   relation, target, target_is_entity, confidence, explicit)` candidates.
2. **Resolve entities**: exact case-insensitive name/alias match against existing entities for
   this user, creating a new `Entity` if nothing matches — no LLM call.
3. **Resolve edge** (deterministic, not LLM): does a current edge already exist for this
   `(source_entity_id, relation)`? If its target differs, close it (`valid_to = now`) and open
   a new one. If it matches, just leave it — no-op. If none exists, open a fresh one. Gated by
   the same explicit-or-`MIN_COMMIT_CONFIDENCE` check as flat facts' safety gate, applied
   inline rather than via the full safety_gate.py machinery (edges don't have a
   category/observation-count shape to gate on).
4. **Write**: the new edge's `fact_sentence` (see Tier 2 above) is embedded and stored.

Both pipelines run for every turn when configured, independent of each other.

**Batched, across both pipelines:**
6. **Reflection** (e.g. hourly per active user, never per-turn): cluster recent writes,
   synthesize a Tier 3 summary where a pattern has emerged across ≥N observations.
7. **Decay sweep** (e.g. daily): old, unreinforced, unretrieved Tier 2 vector rows move to
   Archival. This keeps the active HNSW index small, which is what keeps read-path search fast
   as the system ages — decay and speed are the same mechanism. Graph edges are not decayed:
   closed edges (`valid_to IS NOT NULL`) are permanent history, not a duplicate of the vector
   store to prune.

## 6. Storage

- **Postgres**: `episodes` (raw, append-only, embedded, source of truth — episodic memory, see
  Section 3) · `memory_facts` (Tier 2 vector rows) · `entities` / `memory_edges` (Tier 2
  bi-temporal graph, see Section 3 — `memory_edges.embedding` is the per-edge fact-sentence
  vector, indexed for both similarity search and, via a partial index, fast `valid_to IS NULL`
  current-truth lookups) · `reflections` (Tier 3) · `archival_*` (cold copies) ·
  `expert_identities` / `person_identities` (identity, see Section 3) · `reminders`
  (prospective memory, see Section 3 — no embedding column, indexed on
  `(user_id, status, due_at)` for the due-reminders lookup) — all under one required,
  host-chosen schema (`MemoryConfig.postgres_schema`), never a default `public`. pgvector +
  HNSW index for vector search (used by `memory_facts`, `episodes`, and `memory_edges`).
  Plain indexed edges table with
  recursive CTEs for 1–2 hop graph queries — no separate graph database at this scale.
- **Redis**: Tier 0 session cache, Tier 1 profile cache, durable job stream (Redis Streams)
  feeding the formation worker pool.
- **Formation workers**: a separate deployable from the API, scaled independently, so a
  restart never silently drops queued learning work.

## 7. Non-negotiables

- **User-facing visibility/control**: view, edit, delete stored memories. Both ChatGPT and
  Claude treat this as core product surface, not an afterthought — it also doubles as the
  primary debugging tool during development.
- **Observability**: `logging.getLogger("memory_verse_avneesh.read")` /
  `.formation` — no `print`, ever. `read_memory()` logs what it retrieved from every tier/store
  it queried (Tier 0/1, facts, episodes, edges, identity, due reminders) plus, once retrieval
  finishes, the assembled memory-derived prompt on its own — the read-only portion, before the
  host's live message is appended (see `render_context_as_text`). `write_memory()` logs every
  ADD/UPDATE/DELETE/NOOP, episode write, and edge write/closure with its reasoning. This is the
  only way to see the system's judgment after the fact, since none of it is visible in the final
  response. Configure handlers/level the standard `logging` way — the library attaches none of
  its own.
- **Resilience**: a single backend failing never takes down the whole call.
  `read_memory()` degrades gracefully — each tier/store is retrieved independently, any
  exception is logged in full (`logger.exception`, with traceback) and that piece comes back
  empty rather than raising; the rest of the context is unaffected.
  `write_memory()` is best-effort with per-store *and* per-candidate isolation — one store
  failing (or one bad candidate within a store) doesn't stop the others, each is wrapped and
  logged independently. The one exception: `write_memory()`'s `ValueError` for a caller-contract
  violation (e.g. `graph_store` passed without `relation_extraction_client`) still raises
  immediately — that's a programming error to fix, not a backend failure to degrade around.
- **No implicit credentials**: the library never reads credentials from the environment inside
  its own core code paths — only `MemoryConfig.from_env()` does, and only because a host
  explicitly opted into that convenience by calling it.
- **Per-user isolation**: all storage and queue partitioning keyed by `user_id`, so one user's
  write load never contends with another's reads.

## 8. Build order

Do not build all tiers at once. Per production precedent (Mem0/Zep's own staged rollouts):

**Phase 1 (MVP)**
- Tier 0 (session cache) + Tier 1 (core profile)
- Tier 2, vector half (flat facts + embeddings)
- Tier 2, graph half (entities + bi-temporal edges, fact-sentence embedding retrieval)
- Identity, episodic, and prospective memory
- Formation pipeline: extract → resolve → ADD/UPDATE/DELETE/NOOP → safety gate
- Basic decay sweep
- User-facing memory view/edit/delete

This alone should deliver the large majority of the latency and accuracy win.

**Phase 2**
- Keyword/BM25 channel alongside the vector and graph channels
- Tier 3 reflections
- Full observability/audit trail
- Fuzzy/embedding-based entity resolution (currently exact name/alias match only)

Graduate to Phase 2 only once real usage data from Phase 1 shows where flat-vector retrieval
is actually falling short — not speculatively upfront.

## 9. Packaging: distributed as a PyPI library

**Repo layout — `src` layout (standard for publishable packages, avoids accidentally testing
against the working directory instead of the installed package):**

```
memory-verse-avneesh/                        (repo root)
├── pyproject.toml                   (PEP 621 metadata, build backend, optional-dependencies)
├── README.md
├── LICENSE
├── src/
│   └── memory_verse_avneesh/                (importable package — the actual library)
│       ├── __init__.py              (package metadata; the callables live in read/, formation/, management.py)
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
  dependency: e.g. `pip install memory-verse-avneesh[postgres,redis,bedrock]`.
- **Semantic versioning** from the first published release, since a public API surface means
  breaking changes have real downstream cost.
- `examples/fastapi_app` is a reference/demo of integrating the library into a service — it is
  not part of the published package.

## 10. Open decisions (to confirm before/while building)

- LLM provider and model for extraction and operation-classification calls. (Generation is
  host-owned, not a library decision — see Section 4.)
- Embedding model: local (e.g. sentence-transformers, in-process) vs. hosted API — local
  avoids a network hop on the one embedding call that sits on the critical path.
- Deployment target for the formation worker pool (separate process vs. separate service).
- ~~PyPI package name~~ — decided: `memory-verse-avneesh` (repo renamed to match), import name
  `memory_verse_avneesh`. Third package in the `-verse-avneesh` family alongside
  `storage-verse-avneesh` and `llm-verse-avneesh`.
- **Minimum supported Python version** and how far back to support (affects typing syntax,
  async features available).
