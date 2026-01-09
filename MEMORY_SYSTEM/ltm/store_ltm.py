# MEMORY_SYSTEM/ltm/store_ltm.py

from typing import List, Dict
from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.memory.embeddings import generate_embedding


async def store_ltm_facts(
    user_id: str,
    extracted_facts: List[Dict],
    raw_context: str
) -> None:
    print("\n💾 [LTM-STORE] Starting storage")

    if not extracted_facts:
        print("ℹ️ [LTM-STORE] No facts to store")
        return

    try:
        pool = await db_manager.get_pool()

        async with pool.acquire() as conn:
            for item in extracted_facts:
                fact = item.get("fact")
                memory_type = item.get("memory_type")
                confidence = item.get("confidence_score", 0.0)
                semantic_topic = item.get("semantic_topic")

                if not fact or not memory_type:
                    print("⚠️ [LTM-STORE] Skipping invalid fact:", item)
                    continue

                print(f"\n📝 [LTM-STORE] Processing: {fact}")

                # ------------------------------------
                # Check for existing memory
                # ------------------------------------
                row = await conn.fetchrow(
                    """
                    SELECT memory_id
                    FROM agentic_memory_schema.memories
                    WHERE user_id = $1 AND fact = $2
                    """,
                    user_id,
                    fact
                )

                if row:
                    memory_id = row["memory_id"]
                    print(f"🔁 [LTM-STORE] Reinforcing memory {memory_id}")

                    await conn.execute(
                        """
                        UPDATE agentic_memory_schema.memories
                        SET
                            evidence_count = evidence_count + 1,
                            last_seen_at = NOW()
                        WHERE memory_id = $1
                        """,
                        memory_id
                    )

                else:
                    print("🆕 [LTM-STORE] Creating new memory")

                    embedding = await generate_embedding(fact)

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
                            created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, 'explicit', 'active', $6, NOW())
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
                    print(f"✅ [LTM-STORE] Memory stored: {memory_id}")

                # ------------------------------------
                # Insert memory event
                # ------------------------------------
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
                    VALUES ($1, 'observed', 'extraction', $2, $3, NOW())
                    """,
                    memory_id,
                    confidence,
                    raw_context[:500]
                )

                print("📚 [LTM-STORE] Event logged")

        print("\n🎉 [LTM-STORE] Storage completed successfully")

    except Exception as e:
        print(f"❌ [LTM-STORE] Storage failed: {e}")
