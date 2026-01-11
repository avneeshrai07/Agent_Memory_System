from typing import List, Dict
import traceback

from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.embeddings.encoder import create_embedding

# -------------------------------
# Tunables (aligned with new LTM)
# -------------------------------
SEMANTIC_DUP_DISTANCE = 0.12     # cosine distance threshold
IMPORTANCE_INCREMENT = 0.5
MAX_IMPORTANCE = 10.0


# -------------------------------------------------
# pgvector serialization helper (CRITICAL)
# -------------------------------------------------
def to_pgvector_literal(vec: List[float]) -> str:
    """
    Convert a list[float] into a pgvector-compatible literal string.
    Example: [0.1, -0.2] -> "[0.100000,-0.200000]"
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


async def store_ltm_facts(
    user_id: str,
    extracted_facts: List[Dict],
    raw_context: str
) -> None:
    """
    Store extracted facts into Long-Term Memory with:
    - semantic deduplication
    - frequency & importance reinforcement
    - event logging
    """

    print("\n💾 [LTM] Starting LTM storage")

    if not extracted_facts:
        print("ℹ️ [LTM] No extracted facts provided")
        return

    # -------------------------------------------------
    # 1. Prepare items (NO DB yet)
    # -------------------------------------------------
    prepared_items = []

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
                print(f"⚠️ [LTM] Skipping invalid fact at index {idx}: {item}")
                continue

            print(f"🧠 [LTM] Embedding fact: {fact}")

            embedding = await create_embedding(fact)

            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            if not isinstance(embedding, list):
                raise TypeError(f"Embedding must be list before serialization, got {type(embedding)}")

            embedding = to_pgvector_literal(embedding)

            prepared_items.append({
                "fact": fact,
                "category": category,
                "topic": topic,
                "importance": importance,
                "confidence_score": confidence_score,
                "confidence_source": confidence_source,
                "embedding": embedding
            })

        except Exception:
            print(f"❌ [LTM] Failed preparing fact at index {idx}")
            traceback.print_exc()
            continue

    if not prepared_items:
        print("ℹ️ [LTM] No valid items after preparation")
        return

    # -------------------------------------------------
    # 2. DB Operations
    # -------------------------------------------------
    try:
        pool = await db_manager.get_pool()
    except Exception:
        print("❌ [LTM] Failed to acquire DB pool")
        traceback.print_exc()
        return

    async with pool.acquire() as conn:
        for item in prepared_items:
            fact = item["fact"]
            embedding = item["embedding"]
            category = item["category"]
            topic = item["topic"]
            importance = item["importance"]
            confidence_score = item["confidence_score"]
            confidence_source = item["confidence_source"]

            print(f"\n🧠 [LTM] Processing DB ops for: {fact}")

            # -----------------------------------------
            # 2.1 Semantic deduplication
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
                      AND status = 'active'
                    ORDER BY embedding <-> $2::vector
                    LIMIT 1
                    """,
                    user_id,
                    embedding
                )
            except Exception:
                print("❌ [LTM] Deduplication query failed")
                traceback.print_exc()
                continue

            # -----------------------------------------
            # 2.2 Reinforce existing memory
            # -----------------------------------------
            if row and row["distance"] < SEMANTIC_DUP_DISTANCE:
                try:
                    memory_id = row["memory_id"]
                    new_importance = min(
                        row["importance"] + IMPORTANCE_INCREMENT,
                        MAX_IMPORTANCE
                    )

                    print(
                        f"🔁 [LTM] Reinforcing memory {memory_id} "
                        f"(distance={row['distance']:.4f})"
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
                        memory_id,
                        new_importance
                    )

                except Exception:
                    print("❌ [LTM] Reinforcement update failed")
                    traceback.print_exc()
                    continue

            # -----------------------------------------
            # 2.3 Insert new memory
            # -----------------------------------------
            else:
                try:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO agentic_memory_schema.memories (
                            user_id,
                            category,
                            topic,
                            fact,
                            importance,
                            confidence_score,
                            confidence_source,
                            status,
                            embedding,
                            metadata,
                            created_at,
                            last_updated
                        )
                        VALUES (
                            $1, $2, $3, $4, $5,
                            $6, $7,
                            'active',
                            $8::vector,
                            '{}',
                            NOW(),
                            NOW()
                        )
                        RETURNING memory_id
                        """,
                        user_id,
                        category,
                        topic,
                        fact,
                        importance,
                        confidence_score,
                        confidence_source,
                        embedding
                    )

                    memory_id = row["memory_id"]

                except Exception:
                    print("❌ [LTM] Memory insert failed")
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
                    confidence_score,
                    raw_context[:500]
                )

            except Exception:
                print("❌ [LTM] Event logging failed")
                traceback.print_exc()
                continue

    print("\n🎉 [LTM] Storage completed successfully")
