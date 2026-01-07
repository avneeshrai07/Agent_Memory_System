
# orchestrator.py
"""
Complete memory system orchestration - all layers in one flow.
Triggered after agent generates a response.
"""

from typing import Dict, Any
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.EXTRACTOR.LAYER_1_llm.llm_extractor_prompt import extract_facts_from_conversation
from MEMORY_SYSTEM.EXTRACTOR.LAYER_2_Postgres.layer2_embeddings import create_embeddings
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.preference_patterns import detect_preference_patterns
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.domain_patterns import detect_domain_patterns
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.constraint_patterns import detect_constraint_patterns
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.persona_detector import detect_user_persona
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.load_peronsa import load_user_persona
from MEMORY_SYSTEM.CONSOLIDATION.consolidate_memories import consolidate_memories

# ============================================================================
# PHASE 1: REQUEST ENTRY (BEFORE agent response)
# ============================================================================
async def build_context_with_persona(user_id: str, user_message: str) -> Dict[str, Any]:
    """
    Load user persona and build system prompt BEFORE calling LLM.
    
    Args:
        user_id: User identifier
        user_message: Incoming user message
        
    Returns:
        dict: {system_prompt, persona_metadata}
    """
    # Load active persona
    persona = await load_user_persona(user_id)
    
    # Build adapted system prompt
    if persona:
        metadata = persona['metadata']
        print("persona['metadata']:     ",metadata)
        
        
        
    
    return persona['metadata'] if persona else None


# ============================================================================
# PHASE 2: AFTER AGENT RESPONSE (memory extraction + storage)
# ============================================================================
async def process_conversation_memory(
    user_id: str,
    user_message: str,
    agent_response: str
) -> Dict[str, Any]:
    """
    Complete memory processing pipeline after agent response.
    """
    
    # ========================================================================
    # LAYER 1: EXTRACT FACTS (LLM-based extraction)
    # ========================================================================
    print(f"🔍 [Layer 1] Extracting facts from conversation...")
    fact_extraction_result = await extract_facts_from_conversation(
        user_message=user_message,
        agent_response=agent_response
    )
    
    # ========================================================================
    # FIX: Handle both dict and Pydantic model returns
    # ========================================================================
    if isinstance(fact_extraction_result, dict):
        # If it's a dict, extract the 'facts' key
        facts = fact_extraction_result.get('facts', [])
        print(f"fact_extraction_result (dict): {fact_extraction_result}")
    else:
        # If it's a Pydantic model, access .facts attribute
        facts = fact_extraction_result.facts
    
    if not facts:
        print("   ℹ️  No facts extracted, skipping memory storage")
        return {
            "facts_count": 0,
            "memories_inserted": [],
            "patterns_detected": {},
            "persona_updated": False
        }
    
    print(f"   ✅ Extracted {len(facts)} facts")
    
    # ========================================================================
    # LAYER 2: GENERATE EMBEDDINGS + STORE IN DATABASE
    # ========================================================================
    print(f"🔢 [Layer 2] Generating embeddings and storing memories...")
    pool = await db_manager.wait_for_connection_pool_pool()
    
    # Convert to dicts (handle both ExtractedFact objects and dicts)
    facts_dict = []
    for fact in facts:
        if isinstance(fact, dict):
            # Already a dict
            facts_dict.append({
                "topic": fact.get("topic"),
                "fact": fact.get("fact"),
                "category": fact.get("category"),
                "importance": fact.get("importance")
            })
        else:
            # Pydantic model - use attributes
            facts_dict.append({
                "topic": fact.topic,
                "fact": fact.fact,
                "category": fact.category,
                "importance": fact.importance
            })
    
    inserted_ids = await create_embeddings(pool, user_id, facts_dict)
    print(f"   ✅ Stored {len(inserted_ids)} memories in database")
    
    # ========================================================================
    # LAYER 3: DETECT PATTERNS (run if enough memories accumulated)
    # ========================================================================
    print(f"📊 [Layer 3] Detecting patterns...")
    
    # Check if user has enough memories to detect patterns
    async with pool.acquire() as conn:
        memory_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM agent_memory_system.memories
            WHERE user_id = $1 AND status = 'active';
        """, user_id)
    
    patterns_detected = {}
    persona_updated = False
    
    if memory_count >= 5:  # Threshold: need at least 5 memories
        patterns_detected = await detect_all_patterns(user_id)
        print(f"   ✅ Patterns detected: {len(patterns_detected.get('preferences', []))} preferences, "
              f"{len(patterns_detected.get('domains', []))} domains, "
              f"{len(patterns_detected.get('constraints', []))} constraints")
        
        # ====================================================================
        # LAYER 3.5: BUILD/UPDATE PERSONA (if patterns exist)
        # ====================================================================
        if patterns_detected.get('preferences') or patterns_detected.get('domains'):
            print(f"👤 [Layer 3+] Building user persona...")
            persona_result = await detect_user_persona(user_id)
            
            if persona_result:
                print(f"   ✅ Persona updated: {persona_result['pattern_name']}")
                persona_updated = True
    else:
        print(f"   ℹ️  Insufficient memories ({memory_count}/5), skipping pattern detection")
    
    print(f"""
        "facts_count": {len(facts)},
        "memories_inserted": {inserted_ids},
        "patterns_detected": {patterns_detected},
        "persona_updated": {persona_updated}
          """)
    

    # await background_consolidation_job()
    return {
        "facts_count": len(facts),
        "memories_inserted": inserted_ids,
        "patterns_detected": patterns_detected,
        "persona_updated": persona_updated
    }


# ============================================================================
# HELPER: Detect all patterns (atomic patterns)
# ============================================================================
async def detect_all_patterns(user_id: str) -> dict:
    """
    Run all pattern detectors and return aggregated results.
    """
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        preferences = await detect_preference_patterns(conn, user_id)
        domains = await detect_domain_patterns(conn, user_id)
        constraints = await detect_constraint_patterns(conn, user_id)
        
        return {
            "preferences": preferences,
            "domains": domains,
            "constraints": constraints
        }


# ============================================================================
# USAGE EXAMPLE: Integration with your chat handler
# ============================================================================
async def handle_user_message(user_id: str, user_message: str) -> str:
    """
    Complete message handling flow with persona adaptation.
    
    Usage in your FastAPI route or chat handler:
        response = await handle_user_message(user_id, user_message)
    """
    
    # ========================================================================
    # STEP 1: Load persona and build context (BEFORE LLM call)
    # ========================================================================
    context = await build_context_with_persona(user_id, user_message)
    system_prompt = context['system_prompt']
    
    # ========================================================================
    # STEP 2: Call your "Normal chat llm" with adapted prompt
    # ========================================================================
    # Replace this with your actual LLM call
    agent_response = await your_normal_chat_llm(
        system_prompt=system_prompt,
        user_message=user_message
    )
    
    # ========================================================================
    # STEP 3: Process memory (AFTER agent response) - ASYNC in background
    # ========================================================================
    # This runs in background, doesn't block user response
    import asyncio
    asyncio.create_task(
        process_conversation_memory(
            user_id=user_id,
            user_message=user_message,
            agent_response=agent_response
        )
    )
    
    return agent_response


# ============================================================================
# PLACEHOLDER: Your LLM function (replace with actual implementation)
# ============================================================================
async def your_normal_chat_llm(system_prompt: str, user_message: str) -> str:
    """
    Replace this with your actual LLM call (OpenAI, Bedrock, etc.).
    
    Example with Bedrock:
        response = await bedrock_llm_with_parser(
            system_role=system_prompt,
            prompt=user_message,
            output_schema=YourResponseSchema
        )
    """
    # Placeholder - replace with your implementation
    return f"[Agent response to: {user_message}]"


# ============================================================================
# BACKGROUND JOB: Periodic consolidation + pattern refresh
# ============================================================================
async def background_consolidation_job():
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        users = await conn.fetch("""
            SELECT DISTINCT user_id 
            FROM agent_memory_system.memories
            WHERE status = 'active';
        """)

        for user in users:
            user_id = user['user_id']
            print(f"🔄 Running consolidation for {user_id}...")

            result = await consolidate_memories(conn, user_id)
            print(f"   ✅ Consolidation merged {result['merged']} memories")

            patterns = await detect_all_patterns(user_id)
            if patterns.get('preferences') or patterns.get('domains'):
                await detect_user_persona(user_id)

            print(f"   ✅ Consolidation complete for {user_id}")