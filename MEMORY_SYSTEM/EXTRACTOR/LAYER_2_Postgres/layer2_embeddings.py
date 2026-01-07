# layer2_embeddings_sentence_transformers.py
from sentence_transformers import SentenceTransformer
import asyncpg
import numpy as np
from typing import List, Dict, Any
from pgvector.asyncpg import register_vector
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.DATABASE.INSERT.upsert_level_2_memories import insert_memories
print("Loading GTE-Large (2s first time)...")
model = SentenceTransformer('thenlper/gte-large')  # MTEB 63.13, 1024 dims
print("✅ GTE-Large ready!")


async def create_embeddings(pool, user_id: str, facts: List[Dict]):
    texts = [fact["fact"] for fact in facts]
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    inserted_ids = []
    async with pool.acquire() as conn:
        # REGISTER VECTOR TYPE (sync func in async context)
        await register_vector(conn)  # ← No await!
        
        async with conn.transaction():
            for fact, embedding in zip(facts, embeddings):
                print("fact:    ",fact)
                result = await insert_memories(conn, user_id, fact, embedding.tolist())
                inserted_ids.append(result["id"])
    
    return inserted_ids