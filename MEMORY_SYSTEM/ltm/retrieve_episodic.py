# MEMORY_SYSTEM/ltm/retrieve_episodic.py

from typing import List, Dict
from datetime import datetime
import traceback

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def retrieve_episodic_context(
    user_id: str,
    query_chunks: List[str] | None = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Retrieve active episodic LTM.

    Design guarantees:
    - Episodic memory is ALWAYS loaded (state, not knowledge)
    - Expired episodic memory is excluded
    - Chunk similarity is used ONLY for ordering, never gating
    - No episodic vs factual competition
    """

    try:
        pool = await db_manager.get_pool()
    except Exception:
        traceback.print_exc()
        return []

    now = datetime.utcnow()

    # -------------------------------------------------
    # 1️⃣ Load ALL active episodic memory
    # -------------------------------------------------
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    memory_id,
                    category,
                    topic,
                    fact,
                    confidence_score,
                    metadata,
                    created_at
                FROM agentic_memory_schema.memories
                WHERE user_id = $1
                  AND memory_kind = 'episodic'
                  AND (expires_at IS NULL OR expires_at > $2)
                ORDER BY created_at DESC
                """,
                user_id,
                now,
            )
    except Exception:
        traceback.print_exc()
        return []

    episodic = [dict(r) for r in rows]

    if not episodic:
        return []

    # -------------------------------------------------
    # 2️⃣ Optional chunk-based ordering (SELECTION ONLY)
    # -------------------------------------------------
    if query_chunks:
        for e in episodic:
            score = 0.0
            fact_text = (e.get("fact") or "").lower()

            for chunk in query_chunks:
                if chunk.lower() in fact_text:
                    score += 1.0

            # Episodic score is advisory only
            e["_episodic_score"] = score

        episodic.sort(
            key=lambda x: (
                x.get("_episodic_score", 0.0),
                x.get("confidence_score", 0.0),
                x.get("created_at"),
            ),
            reverse=True,
        )

        # Clean internal scoring field
        for e in episodic:
            e.pop("_episodic_score", None)

    # -------------------------------------------------
    # 3️⃣ Hard cap (defensive)
    # -------------------------------------------------
    return episodic[:limit]
