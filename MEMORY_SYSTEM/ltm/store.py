# MEMORY_SYSTEM/ltm/store.py

from MEMORY_SYSTEM.embeddings.encoder import embed

async def store_fact(conn, user_id: str, fact: dict):
    vector = embed(fact["fact"])

    await conn.execute("""
        INSERT INTO agentic_memory_schema.ltm_memories (
            user_id,
            topic,
            fact,
            category,
            signal_type,
            importance,
            embedding
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7)
    """,
        user_id,
        fact["topic"],
        fact["fact"],
        fact["category"],
        fact["signal_type"],
        fact["importance"],
        vector
    )
