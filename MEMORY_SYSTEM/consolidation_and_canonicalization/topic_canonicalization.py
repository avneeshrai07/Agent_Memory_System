from typing import Dict, List
import asyncpg


async def canonicalize_topics(
    conn,
    user_id: str,
):
    """
    Level-2 consolidation:
    - Ensures ONE active memory per (memory_type, semantic_topic)
    - Demotes others to status='supporting'
    - Fully reversible and auditable
    """

    try:
        async with conn.transaction():

            # 1. Find topics with multiple active memories
            topic_rows = await conn.fetch(
                """
                SELECT
                    memory_type,
                    semantic_topic,
                    COUNT(*) AS cnt
                FROM agentic_memory_schema.memories
                WHERE user_id = $1
                  AND status = 'active'
                  AND semantic_topic IS NOT NULL
                GROUP BY memory_type, semantic_topic
                HAVING COUNT(*) > 1;
                """,
                user_id,
            )

            if not topic_rows:
                return {
                    "canonicalized": 0,
                    "details": [],
                }

            canonicalized_count = 0
            details = []

            for row in topic_rows:
                memory_type = row["memory_type"]
                semantic_topic = row["semantic_topic"]

                # 2. Fetch all active memories for this topic
                memories = await conn.fetch(
                    """
                    SELECT
                        memory_id,
                        confidence_score,
                        evidence_count,
                        last_seen_at
                    FROM agentic_memory_schema.memories
                    WHERE user_id = $1
                      AND status = 'active'
                      AND memory_type = $2
                      AND semantic_topic = $3;
                    """,
                    user_id,
                    memory_type,
                    semantic_topic,
                )

                if len(memories) < 2:
                    continue

                # 3. Choose canonical
                canonical = max(
                    memories,
                    key=lambda m: (
                        m["confidence_score"],
                        m["evidence_count"],
                        m["last_seen_at"],
                    ),
                )

                canonical_id = canonical["memory_id"]

                supporting_ids = [
                    m["memory_id"]
                    for m in memories
                    if m["memory_id"] != canonical_id
                ]

                # 4. Demote others (non-destructive)
                await conn.execute(
                    """
                    UPDATE agentic_memory_schema.memories
                    SET status = 'supporting'
                    WHERE memory_id = ANY($1);
                    """,
                    supporting_ids,
                )

                canonicalized_count += len(supporting_ids)

                details.append(
                    {
                        "memory_type": memory_type,
                        "semantic_topic": semantic_topic,
                        "canonical_id": str(canonical_id),
                        "supporting_count": len(supporting_ids),
                    }
                )

            return {
                "canonicalized": canonicalized_count,
                "details": details,
            }

    except asyncpg.PostgresError as db_error:
        return {
            "canonicalized": 0,
            "error_type": "database_error",
            "error": str(db_error),
        }

    except Exception as e:
        return {
            "canonicalized": 0,
            "error_type": "unexpected_error",
            "error": str(e),
        }
