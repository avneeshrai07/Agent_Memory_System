from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.persona_shift import detect_persona_shift
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.upsert_user_persona import upsert_user_pattern
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.persona_detector import detect_user_persona
async def recalibrate_all_personas():
    """
    Monthly job: rebuild personas from updated patterns.
    """
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        users = await conn.fetch("""
            SELECT DISTINCT user_id 
            FROM agent_memory_system.user_patterns
            WHERE status = 'active';
        """)
        
        for user in users:
            user_id = user['user_id']
            
            # Detect persona shift
            await detect_persona_shift(conn, user_id)
            
            # Rebuild persona from current patterns
            persona = await detect_user_persona(conn, user_id)
            await upsert_user_pattern(conn, user_id, persona)
            
            print(f"✅ Persona recalibrated for {user_id}")
