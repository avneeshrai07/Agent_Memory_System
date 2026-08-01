# FastAPI reference integration

Shows how a host application wires `agent_memory` in. Ships with the repo
for reference — it is not part of the published `agent-memory` package.

## Run

```bash
pip install -e ".[postgres,redis,bedrock]"
pip install -r examples/fastapi_app/requirements.txt
cp examples/fastapi_app/.env.example examples/fastapi_app/.env  # fill in values
uvicorn examples.fastapi_app.main:app --reload
```

Requires a reachable Postgres (with the `vector` extension available) and
Redis, and AWS credentials with Bedrock access in `AWS_REGION`.

## Try it

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "conversation_id": "c1", "message": "I always want short, bullet-pointed answers"}'
```

Send a second message in the same conversation a few seconds later — the
first turn's formation pass should have run by then — and ask something
where that preference would matter, to see it come back through Tier 2.

View, correct, or remove what got stored:

```bash
curl http://127.0.0.1:8000/memories/u1
curl -X PATCH http://127.0.0.1:8000/memories/<fact_id> -H "Content-Type: application/json" -d '{"value": "corrected value"}'
curl -X DELETE http://127.0.0.1:8000/memories/<fact_id>
```

Run the decay sweep (batched maintenance — run this on a schedule, not per-request):

```bash
python -m examples.fastapi_app.run_decay_sweep
```

## What this does and doesn't demonstrate

Does: the full retrieval loop (Tier 0/1/2 reads, concurrent, no LLM call);
the full formation loop (extract → resolve → classify ADD/UPDATE/DELETE/NOOP
→ safety gate → write); user-facing view/edit/delete; and the decay sweep —
all wired against real Postgres/Redis/Bedrock. It also demonstrates the
actual contract: `agent_memory` never generates the response. `main.py`'s
`/chat` handler calls `read_memory()` (library), then makes its own Bedrock
call via `llm.py`'s `generate_response()` (plain host code, not part of the
package), then builds the `Turn` itself and hands it to `write_memory()`.
Swap `llm.py` for tool-calling, streaming, a different model, or a
different provider entirely — the library doesn't know or care.

Doesn't: durable formation delivery. This example hands the completed turn
to `write_memory()` via FastAPI's `BackgroundTasks`, which runs after the
response is sent but is lost if the process dies first. See README Section
6/9 in the repo root for the production pattern (durable queue + separate
worker pool) — this example intentionally stays simple.
