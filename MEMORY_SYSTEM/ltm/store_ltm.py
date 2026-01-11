from typing import List, Dict
import traceback
import json
from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.embeddings.encoder import create_embedding


# -------------------------------
# Tunables
# -------------------------------
SEMANTIC_DUP_DISTANCE = 0.12
IMPORTANCE_INCREMENT = 0.5
MAX_IMPORTANCE = 10.0


def to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


async def store_ltm_facts(
    user_id: str,
    extracted_facts: List[Dict],
    raw_context: str
) -> None:

    print("\n💾 [LTM] Starting LTM storage")

    if not extracted_facts:
        print("ℹ️ [LTM] No extracted facts provided")
        return

    prepared_items = []

    # -------------------------------------------------
    # 1️⃣ Prepare items
    # -------------------------------------------------
    for idx, item in enumerate(extracted_facts):
        try:
            fact = item.get("fact")
            category = item.get("category")
            topic = item.get("topic")
            importance = float(item.get("importance", 5.0))

            confidence = item.get("confidence", {})
            confidence_score = float(confidence.get("score", 0.7))
            confidence_source = confidence.get("source", "implicit")

            if not fact or not category or not topic:
                continue

            embedding = await create_embedding(fact)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            embedding = to_pgvector_literal(embedding)

            prepared_items.append({
                "fact": fact,
                "category": category,
                "topic": topic,
                "importance": importance,
                "confidence_score": confidence_score,
                "confidence_source": confidence_source,
                "embedding": embedding,
                "metadata": json.dumps({}),   # ✅ explicit
            })

        except Exception:
            traceback.print_exc()
            continue

    if not prepared_items:
        return

    # -------------------------------------------------
    # 2️⃣ DB operations
    # -------------------------------------------------
    pool = await db_manager.get_pool()

    async with pool.acquire() as conn:
        for item in prepared_items:
            fact = item["fact"]

            print(f"\n🧠 [LTM] Processing DB ops for: {fact}")

            # -----------------------------------------
            # 2.1 Deduplication (FACTUAL ONLY)
            # -----------------------------------------
            try:
                row = await conn.fetchrow(
                    """
                    SELECT
                        memory_id,
                        importance,
                        embedding <-> $2::vector AS distance
                    FROM agentic_memory_schema.memories
                    WHERE user_id = $1
                      AND memory_kind = 'factual'
                      AND status = 'active'
                    ORDER BY embedding <-> $2::vector
                    LIMIT 1
                    """,
                    user_id,
                    item["embedding"],
                )
            except Exception:
                traceback.print_exc()
                continue

            # -----------------------------------------
            # 2.2 Reinforce
            # -----------------------------------------
            if row and row["distance"] < SEMANTIC_DUP_DISTANCE:
                try:
                    new_importance = min(
                        row["importance"] + IMPORTANCE_INCREMENT,
                        MAX_IMPORTANCE
                    )

                    await conn.execute(
                        """
                        UPDATE agentic_memory_schema.memories
                        SET
                            frequency = frequency + 1,
                            importance = $2,
                            last_updated = NOW()
                        WHERE memory_id = $1
                        """,
                        row["memory_id"],
                        new_importance,
                    )

                    memory_id = row["memory_id"]

                except Exception:
                    traceback.print_exc()
                    continue

            # -----------------------------------------
            # 2.3 Insert new factual memory
            # -----------------------------------------
            else:
                try:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO agentic_memory_schema.memories (
                            user_id,
                            memory_kind,
                            category,
                            topic,
                            fact,
                            importance,
                            confidence_score,
                            confidence_source,
                            frequency,
                            status,
                            embedding,
                            metadata,
                            created_at,
                            last_updated
                        )
                        VALUES (
                            $1,
                            'factual',
                            $2,
                            $3,
                            $4,
                            $5,
                            $6,
                            $7,
                            1,
                            'active',
                            $8,
                            $9,
                            NOW(),
                            NOW()
                        )
                        RETURNING memory_id
                        """,
                        user_id,
                        item["category"],
                        item["topic"],
                        item["fact"],
                        item["importance"],
                        item["confidence_score"],
                        item["confidence_source"],
                        item["embedding"],
                        item["metadata"],
                    )

                    memory_id = row["memory_id"]

                except Exception:
                    traceback.print_exc()
                    continue

            # -----------------------------------------
            # 2.4 Memory event log
            # -----------------------------------------
            try:
                await conn.execute(
                    """
                    INSERT INTO agentic_memory_schema.memory_events (
                        memory_id,
                        event_type,
                        source,
                        signal_strength,
                        raw_context,
                        metadata,
                        created_at
                    )
                    VALUES (
                        $1,
                        'extracted',
                        'llm',
                        $2,
                        $3,
                        '{}',
                        NOW()
                    )
                    """,
                    memory_id,
                    item["confidence_score"],
                    raw_context[:500],
                )
                print("\n🎉 [LTM FACTUAL] Storage completed successfully")
            except Exception:
                traceback.print_exc()
                continue

    
