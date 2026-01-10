from typing import List, Dict
import traceback

from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.embeddings.encoder import create_embedding

# -------------------------------
# Tunables
# -------------------------------
SEMANTIC_DUP_DISTANCE = 0.12     # cosine distance threshold
CONFIDENCE_INCREMENT = 0.05
MAX_CONFIDENCE = 1.0


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
    - confidence evolution
    - evidence logging
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
            memory_type = item.get("memory_type")
            semantic_topic = item.get("semantic_topic")
            confidence = float(item.get("confidence_score", 0.5))

            if not fact or not memory_type:
                print(f"⚠️ [LTM] Skipping invalid fact at index {idx}: {item}")
                continue

            print(f"🧠 [LTM] Embedding fact: {fact}")

            embedding = await create_embedding(fact)

            # ---- OPTION A FIX ----
            # numpy.ndarray -> list -> pgvector literal string
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            if not isinstance(embedding, list):
                raise TypeError(f"Embedding must be list before serialization, got {type(embedding)}")

            embedding = to_pgvector_literal(embedding)
            # ----------------------

            prepared_items.append({
                "fact": fact,
                "memory_type": memory_type,
                "semantic_topic": semantic_topic,
                "confidence": confidence,
                "embedding": embedding,   # <-- STRING now
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
        for idx, item in enumerate(prepared_items):
            fact = item["fact"]
            embedding = item["embedding"]   # <-- STRING
            memory_type = item["memory_type"]
            semantic_topic = item["semantic_topic"]
            confidence = item["confidence"]

            print(f"\n🧠 [LTM] Processing DB ops for: {fact}")

            # -----------------------------------------
            # 2.1 Semantic deduplication
            # -----------------------------------------
            try:
                row = await conn.fetchrow(
                    """
                    SELECT
                        memory_id,
                        confidence_score,
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
                print(f"Embedding type: {type(embedding)}")
                traceback.print_exc()
                continue

            # -----------------------------------------
            # 2.2 Reinforce existing memory
            # -----------------------------------------
            if row and row["distance"] < SEMANTIC_DUP_DISTANCE:
                try:
                    memory_id = row["memory_id"]
                    new_confidence = min(
                        row["confidence_score"] + CONFIDENCE_INCREMENT,
                        MAX_CONFIDENCE
                    )

                    print(
                        f"🔁 [LTM] Reinforcing memory {memory_id} "
                        f"(distance={row['distance']:.4f})"
                    )

                    await conn.execute(
                        """
                        UPDATE agentic_memory_schema.memories
                        SET
                            evidence_count = evidence_count + 1,
                            confidence_score = $2,
                            last_seen_at = NOW()
                        WHERE memory_id = $1
                        """,
                        memory_id,
                        new_confidence
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
                    # print("🆕 [LTM] Creating new memory")

                    row = await conn.fetchrow(
                        """
                        INSERT INTO agentic_memory_schema.memories (
                            user_id,
                            fact,
                            memory_type,
                            semantic_topic,
                            confidence_score,
                            confidence_source,
                            status,
                            embedding,
                            evidence_count,
                            created_at,
                            last_seen_at
                        )
                        VALUES (
                            $1, $2, $3, $4, $5,
                            'explicit',
                            'active',
                            $6::vector,
                            1,
                            NOW(),
                            NOW()
                        )
                        RETURNING memory_id
                        """,
                        user_id,
                        fact,
                        memory_type,
                        semantic_topic,
                        confidence,
                        embedding
                    )

                    memory_id = row["memory_id"]
                    # print(f"✅ [LTM] Stored new memory {memory_id}")

                except Exception:
                    print("❌ [LTM] Memory insert failed")
                    print(f"Embedding type: {type(embedding)}")
                    traceback.print_exc()
                    continue

            # -----------------------------------------
            # 2.4 Evidence / event log
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
                        created_at
                    )
                    VALUES (
                        $1,
                        'observed',
                        'extraction',
                        $2,
                        $3,
                        NOW()
                    )
                    """,
                    memory_id,
                    confidence,
                    raw_context[:500]
                )

                # print("📚 [LTM] Evidence logged")

            except Exception:
                print("❌ [LTM] Evidence logging failed")
                traceback.print_exc()
                continue

    print("\n🎉 [LTM] Storage completed successfully")
