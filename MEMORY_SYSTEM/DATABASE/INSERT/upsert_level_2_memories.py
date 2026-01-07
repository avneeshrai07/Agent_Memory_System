# MEMORY_SYSTEM/DATABASE/INSERT/upsert_level_2_memories.py

async def insert_memories(conn, user_id: str, fact: dict, embedding: list):
    """
    Insert memory into database with detailed logging.
    """
    print(f"📝 Attempting to insert memory:")
    print(f"   user_id: {user_id}")
    print(f"   topic: {fact['topic']}")
    print(f"   category: {fact['category']}")
    print(f"   fact: {fact['fact']}")
    print(f"   importance: {fact['importance']}")
    
    try:
        result = await conn.fetchrow("""
            INSERT INTO agent_memory_system.memories (
                user_id,
                topic,
                fact,
                category,
                importance_score,
                embedding
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, category, topic;
        """,
            user_id,
            fact['topic'],
            fact['fact'],
            fact['category'],
            fact['importance'],
            embedding
        )
        
        print(f"   ✅ SUCCESS: Inserted memory ID={result['id']}, category={result['category']}, topic={result['topic']}")
        return {"id": result['id']}
        
    except Exception as e:
        print(f"   ❌ FAILED TO INSERT:")
        print(f"      Error: {e}")
        print(f"      Category that failed: {fact['category']}")
        print(f"      Topic: {fact['topic']}")
        raise  # Re-raise to see full traceback
