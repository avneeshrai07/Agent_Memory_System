# MEMORY_SYSTEM/EXTRACTOR/LAYER_3_pattern/PATTERN/load_peronsa.py

from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
import json  # ← ADD THIS


async def load_user_persona(user_id: str) -> dict:
    """
    LOADER: Fetches existing persona from database.
    This is a READER - it only queries, never modifies.
    """
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        
        # 1. Query database for existing persona
        persona = await conn.fetchrow("""
            SELECT pattern_name, description, confidence, metadata
            FROM agent_memory_system.user_patterns
            WHERE user_id = $1
              AND pattern_type = 'persona'
              AND status = 'active'
            ORDER BY last_observed DESC
            LIMIT 1;
        """, user_id)
        
        # 2. Return data (or None if doesn't exist)
        if not persona:
            return None
        
        persona_dict = dict(persona)
        
        # 🆕 FIX: Parse metadata if it's a JSON string
        if isinstance(persona_dict['metadata'], str):
            persona_dict['metadata'] = json.loads(persona_dict['metadata'])
        
        return persona_dict
