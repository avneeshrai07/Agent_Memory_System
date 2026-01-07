from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.preference_patterns import detect_preference_patterns
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.domain_patterns import detect_domain_patterns
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.constraint_patterns import detect_constraint_patterns
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.persona_detector import detect_user_persona


async def detect_all_patterns(user_id: str) -> dict:
    """
    Run all pattern detectors and return aggregated results.
    
    Flow:
    1. Detect atomic patterns (preferences, domains, constraints)
    2. Build composite persona from those patterns
    3. Return all detected patterns including persona
    """
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        # ================================================================
        # STEP 1: Detect atomic patterns (base layer)
        # ================================================================
        preferences = await detect_preference_patterns(conn, user_id)
        domains = await detect_domain_patterns(conn, user_id)
        constraints = await detect_constraint_patterns(conn, user_id)
        
        print(f"✅ Detected {len(preferences)} preferences, {len(domains)} domains, {len(constraints)} constraints")
        
        # ================================================================
        # STEP 2: Build persona from detected patterns (composite layer)
        # ================================================================
        persona = None
        if preferences or domains:
            # Only build persona if we have base patterns
            print("👤 Building user persona from detected patterns...")
            persona = await detect_user_persona(user_id)  # This CREATES/UPDATES persona
        else:
            print("⚠️  No base patterns found, skipping persona creation")
        
        # ================================================================
        # STEP 3: Return all detected patterns
        # ================================================================
        return {
            "preferences": preferences,
            "domains": domains,
            "constraints": constraints,
            "persona": persona  # ← Newly created/updated persona
        }
