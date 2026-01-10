from pgvector.asyncpg import register_vector
import asyncpg


async def consolidate_memories(
    conn,
    user_id: str,
    similarity_threshold: float = 0.85,
    candidate_limit: int = 50,
):
    """
    Production-safe memory consolidation with error handling.

    - Uses agentic_memory_schema.memories
    - Fully schema-aligned
    - Transaction-safe
    """

    try:
        # Ensure pgvector is registered
        await register_vector(conn)

        # Use explicit transaction for safety
        async with conn.transaction():

            # 1. Fetch active memories
            rows = await conn.fetch(
                """
                SELECT
                    memory_id,
                    semantic_topic,
                    fact,
                    embedding,
                    confidence_score,
                    evidence_count,
                    memory_type,
                    last_seen_at
                FROM agentic_memory_schema.memories
                WHERE user_id = $1
                  AND status = 'active'
                ORDER BY confidence_score DESC,
                         evidence_count DESC,
                         last_seen_at DESC
                LIMIT $2;
                """,
                user_id,
                candidate_limit,
            )

            if len(rows) < 2:
                return {
                    "merged": 0,
                    "reason": "Not enough memories to consolidate"
                }

            visited = set()
            merged_count = 0
            consolidations = []

            for base in rows:
                base_id = base["memory_id"]

                if base_id in visited:
                    continue

                # 2. Find similar memories (hard partitioning)
                similar = await conn.fetch(
                    """
                    SELECT
                        memory_id,
                        semantic_topic,
                        confidence_score,
                        evidence_count,
                        last_seen_at,
                        1 - (embedding <=> $1::vector) AS similarity
                    FROM agentic_memory_schema.memories
                    WHERE user_id = $2
                      AND status = 'active'
                      AND memory_type = $3
                      AND memory_id != $4
                      AND 1 - (embedding <=> $1::vector) >= $5
                    ORDER BY similarity DESC
                    LIMIT $6;
                    """,
                    base["embedding"],
                    user_id,
                    base["memory_type"],
                    base_id,
                    similarity_threshold,
                    candidate_limit,
                )

                if not similar:
                    continue

                # 3. Canonical selection (strongest wins)
                canonical = max(
                    [base, *similar],
                    key=lambda r: (
                        r["confidence_score"],
                        r["evidence_count"],
                        r["last_seen_at"],
                    ),
                )

                canonical_id = canonical["memory_id"]

                merged_ids = [
                    r["memory_id"]
                    for r in similar
                    if r["memory_id"] != canonical_id
                ]

                if not merged_ids:
                    continue

                # 4. Update canonical memory
                await conn.execute(
                    """
                    UPDATE agentic_memory_schema.memories
                    SET
                        evidence_count = evidence_count + $2,
                        last_seen_at = NOW()
                    WHERE memory_id = $1;
                    """,
                    canonical_id,
                    len(merged_ids),
                )

                # 5. Mark merged memories
                await conn.execute(
                    """
                    UPDATE agentic_memory_schema.memories
                    SET status = 'merged'
                    WHERE memory_id = ANY($1);
                    """,
                    merged_ids,
                )

                visited.add(canonical_id)
                visited.update(merged_ids)
                merged_count += len(merged_ids)

                consolidations.append(
                    {
                        "canonical_id": str(canonical_id),
                        "merged_count": len(merged_ids),
                        "semantic_topic": canonical["semantic_topic"],
                    }
                )

            return {
                "merged": merged_count,
                "consolidations": consolidations,
            }

    except asyncpg.PostgresError as db_error:
        # Database-level errors (constraint violations, syntax, etc.)
        return {
            "merged": 0,
            "error_type": "database_error",
            "error": str(db_error),
        }

    except Exception as unexpected_error:
        # Any other unexpected failure
        return {
            "merged": 0,
            "error_type": "unexpected_error",
            "error": str(unexpected_error),
        }
