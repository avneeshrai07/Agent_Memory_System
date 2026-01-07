# MEMORY_SYSTEM/EXTRACTOR/LAYER_3_pattern/load_patterns.py
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
async def load_all_user_patterns(user_id: str, min_confidence: float = 0.75) -> dict:
    """
    LOADER: Load all active patterns for runtime use (READ-ONLY).
    """
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        # DEBUG: Check what's in the database
        total_patterns = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM agent_memory_system.user_patterns
            WHERE user_id = $1;
        """, user_id)
        
        print(f"🔍 [LOAD] Total patterns for user {user_id}: {total_patterns}")
        
        # Load ALL active patterns (including persona)
        rows = await conn.fetch("""
            SELECT 
                pattern_type, 
                pattern_name, 
                description, 
                confidence,
                frequency,
                status,
                metadata
            FROM agent_memory_system.user_patterns
            WHERE user_id = $1 
              AND status = 'active'
              AND confidence >= $2
            ORDER BY pattern_type, confidence DESC;
        """, user_id, min_confidence)
        
        print(f"🔍 [LOAD] Found {len(rows)} active patterns (min confidence: {min_confidence}):")
        for row in rows:
            print(f"   - {row['pattern_type']}: {row['pattern_name']} (confidence: {row['confidence']}, status: {row['status']})")
        
        # Group by pattern type
        patterns = {
            "preferences": [],
            "domains": [],
            "constraints": [],
            "expertise": [],
            "persona": None
        }
        
        for row in rows:
            pattern_dict = dict(row)
            pattern_type = pattern_dict['pattern_type']
            
            print(f"   🔍 Grouping: type='{pattern_type}', name='{pattern_dict['pattern_name']}'")  # ← DEBUG
            
            if pattern_type == 'persona':
                patterns['persona'] = pattern_dict
                print(f"      ✅ Added to persona")
            elif pattern_type == 'preference':
                patterns['preferences'].append(pattern_dict)
                print(f"      ✅ Added to preferences (total: {len(patterns['preferences'])})")
            elif pattern_type == 'domain':
                patterns['domains'].append(pattern_dict)
                print(f"      ✅ Added to domains (total: {len(patterns['domains'])})")
            elif pattern_type == 'constraint':
                patterns['constraints'].append(pattern_dict)
                print(f"      ✅ Added to constraints (total: {len(patterns['constraints'])})")
            elif pattern_type == 'expertise':
                patterns['expertise'].append(pattern_dict)
                print(f"      ✅ Added to expertise (total: {len(patterns['expertise'])})")
            else:
                print(f"      ⚠️ UNKNOWN pattern_type: '{pattern_type}'")
        
        print(f"🔍 [LOAD] Grouped patterns:")
        print(f"   Preferences: {len(patterns['preferences'])}")
        print(f"   Domains: {len(patterns['domains'])}")
        print(f"   Constraints: {len(patterns['constraints'])}")
        print(f"   Expertise: {len(patterns['expertise'])}")
        print(f"   Persona: {'Yes' if patterns['persona'] else 'No'}")
        
        print("patterns:    ",patterns)
        return patterns
