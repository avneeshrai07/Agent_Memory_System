async def insert_memories(conn, user_id, fact, embedding):
    result = await conn.fetchrow(
        """
        INSERT INTO agent_memory_system.memories (
            user_id, topic, fact, category, importance_score, embedding
        ) VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id;
        """,
        user_id, fact["topic"], fact["fact"], fact["category"], 
        fact["importance"], embedding  # List[float] ✓
    )
    return dict(result)
