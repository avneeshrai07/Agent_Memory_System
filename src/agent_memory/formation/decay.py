"""Decay sweep (README Section 5, step 7).

Batched maintenance — a host application runs this periodically (e.g. daily,
via its own scheduler/cron), separately from the per-turn formation loop.
Moves old, unreinforced facts out of the actively-searched set so Tier 2
retrieval stays fast as memory accumulates: FactStore.search_facts only
ever considers status='active' rows, so archiving is what keeps that set
(and the index behind it) from growing without bound — decay and speed are
the same mechanism here, not just cleanup.

Deliberately reinforcement/age-based only, not "unretrieved": tracking
retrieval recency would mean writing to a fact on every read-path search,
and the read path never writes — that's a foundational constraint of this
library (README Section 2), not an oversight.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent_memory.models import MemoryStatus
from agent_memory.storage.interfaces import FactStore

DEFAULT_DECAY_AFTER = timedelta(days=90)
DEFAULT_BATCH_SIZE = 500


async def run_decay_sweep(
    *,
    fact_store: FactStore,
    decay_after: timedelta = DEFAULT_DECAY_AFTER,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Archives every active/provisional fact not reinforced since
    `decay_after` ago, across all users. Returns the number archived.

    Processes in batches so a large table doesn't require one unbounded
    query — each archived fact drops out of list_decayable_facts's own
    filter, so the loop naturally converges.
    """

    threshold = datetime.now(timezone.utc) - decay_after
    archived_count = 0

    while True:
        batch = await fact_store.list_decayable_facts(threshold, batch_size)
        if not batch:
            break

        for fact in batch:
            archived = fact.model_copy(update={"status": MemoryStatus.ARCHIVED})
            await fact_store.update_fact(archived)
            archived_count += 1

        if len(batch) < batch_size:
            break

    return archived_count
