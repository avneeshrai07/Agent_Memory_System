# MEMORY_SYSTEM/DATABASE/INSERT/upsert_user_pattern.py
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
# MEMORY_SYSTEM/EXTRACTOR/LAYER_3_pattern/upsert_user_persona.py (or wherever it is)

import json  # ← Add this import at the top

async def upsert_user_pattern(
    user_id: str,
    pattern_type: str,
    pattern_name: str,
    confidence: float,
    frequency: int = 1,
    signals_count: int = 1,
    description: str = None,
    metadata: dict = None
):
    """
    Insert or update a user pattern.
    """
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        
        print(f"📝 [UPSERT] Upserting pattern:")
        print(f"   user_id: {user_id}")
        print(f"   type: {pattern_type}")
        print(f"   name: {pattern_name}")
        print(f"   confidence: {confidence}")
        
        # 🆕 FIX: Ensure metadata is JSONB-compatible
        metadata_json = json.dumps(metadata) if metadata else '{}'
        
        result = await conn.fetchrow("""
            INSERT INTO agent_memory_system.user_patterns (
                user_id, pattern_type, pattern_name, description,
                confidence, frequency, signals_count, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            ON CONFLICT (user_id, pattern_type, pattern_name)
            DO UPDATE SET
                confidence = EXCLUDED.confidence,
                frequency = user_patterns.frequency + EXCLUDED.frequency,
                signals_count = user_patterns.signals_count + EXCLUDED.signals_count,
                last_observed = NOW(),
                description = COALESCE(EXCLUDED.description, user_patterns.description),
                metadata = user_patterns.metadata || EXCLUDED.metadata,
                status = 'active'
            RETURNING id, pattern_name, confidence, status;
        """,
            user_id,
            pattern_type,
            pattern_name,
            description,
            confidence,
            frequency,
            signals_count,
            metadata_json  # ← Pass JSON string, not dict
        )
        
        print(f"✅ [UPSERT] Success: ID={result['id']}, name={result['pattern_name']}, confidence={result['confidence']}, status={result['status']}")
        
        return dict(result)

