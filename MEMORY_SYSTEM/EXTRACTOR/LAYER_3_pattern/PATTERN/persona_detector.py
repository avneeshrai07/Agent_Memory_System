# MEMORY_SYSTEM/EXTRACTOR/LAYER_3_pattern/PATTERN/persona_detector.py

from datetime import datetime
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.upsert_user_persona import upsert_user_pattern  # ← ADD THIS


async def detect_user_persona(user_id: str) -> dict:
    """
    Build a composite user persona from detected patterns.
    
    Logic:
    - Aggregates tone, style, template, objective preferences
    - Uses weighted frequency + recency
    - Produces a single "persona" pattern with high confidence
    """
    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:
        # === TONE DETECTION ===
        tone_signals = await conn.fetch("""
            SELECT 
                LOWER(pattern_name) as tone,
                frequency,
                confidence,
                EXTRACT(EPOCH FROM (NOW() - last_observed)) / 86400.0 as days_since
            FROM agent_memory_system.user_patterns
            WHERE user_id = $1
            AND pattern_type = 'preference'
            AND pattern_name IN ('formal_tone', 'casual_tone', 'professional_tone', 'friendly_tone')
            AND status = 'active'
            ORDER BY confidence DESC, frequency DESC
            LIMIT 1;
        """, user_id)
        
        tone = tone_signals[0]['tone'].replace('_tone', '') if tone_signals else 'neutral'
        
        # === STYLE DETECTION (concise vs detailed) ===
        style_signals = await conn.fetch("""
            SELECT 
                CASE 
                    WHEN pattern_name ILIKE '%concise%' OR pattern_name ILIKE '%brief%' THEN 'concise'
                    WHEN pattern_name ILIKE '%detailed%' OR pattern_name ILIKE '%comprehensive%' THEN 'detailed'
                    ELSE 'balanced'
                END as style,
                SUM(frequency) as total_frequency
            FROM agent_memory_system.user_patterns
            WHERE user_id = $1
            AND pattern_type = 'preference'
            AND status = 'active'
            GROUP BY style
            ORDER BY total_frequency DESC
            LIMIT 1;
        """, user_id)
        
        style = style_signals[0]['style'] if style_signals else 'balanced'
        
        # === TEMPLATE TYPE (from domains) ===
        template_signals = await conn.fetch("""
            SELECT pattern_name, frequency
            FROM agent_memory_system.user_patterns
            WHERE user_id = $1
            AND pattern_type = 'domain'
            AND pattern_name IN ('marketing_email', 'technical_doc', 'sales_pitch', 'support_ticket')
            AND status = 'active'
            ORDER BY confidence DESC, frequency DESC
            LIMIT 1;
        """, user_id)
        
        template_type = template_signals[0]['pattern_name'] if template_signals else 'general'
        
        # === OBJECTIVES (from constraint/preference patterns) ===
        objective_signals = await conn.fetch("""
            SELECT 
                ARRAY_AGG(DISTINCT pattern_name) as objectives
            FROM agent_memory_system.user_patterns
            WHERE user_id = $1
            AND pattern_type IN ('constraint', 'preference')
            AND pattern_name IN ('drive_conversions', 'build_trust', 'educate_audience', 'quick_response')
            AND status = 'active';
        """, user_id)
        
        objectives = objective_signals[0]['objectives'] if objective_signals and objective_signals[0]['objectives'] else []
        
        # === TARGET AUDIENCE (extract from memories) ===
        audience_signals = await conn.fetch("""
            SELECT fact
            FROM agent_memory_system.memories
            WHERE user_id = $1
            AND category = 'constraint'
            AND topic ILIKE '%audience%'
            AND status = 'active'
            ORDER BY importance_score DESC
            LIMIT 1;
        """, user_id)
        
        target_audience = audience_signals[0]['fact'] if audience_signals else 'general_audience'
        
        # === BUILD PERSONA METADATA ===
        persona_metadata = {
            "tone": tone,
            "style": style,
            "template_type": template_type,
            "objectives": objectives,
            "target_audience": target_audience,
            "last_updated": datetime.now().isoformat()
        }
        
        # Calculate composite confidence (average of component confidences)
        component_confidence = 0.85  # High confidence for aggregated pattern
        
        pattern_name = f"{tone}_{style}_{template_type}"
        
        print(f"🔍 [PERSONA] Detected persona components:")
        print(f"   tone: {tone}")
        print(f"   style: {style}")
        print(f"   template_type: {template_type}")
        print(f"   objectives: {objectives}")
        print(f"   pattern_name: {pattern_name}")
        print(f"   confidence: {component_confidence}")
        
        # 🆕 CRITICAL: Actually save the persona to the database!
        result = await upsert_user_pattern(
            user_id=user_id,
            pattern_type="persona",
            pattern_name=pattern_name,
            confidence=component_confidence,
            frequency=1,
            signals_count=len(tone_signals) + len(style_signals) + len(template_signals),
            description=f"User prefers {tone} tone with {style} style for {template_type}",
            metadata=persona_metadata
        )
        
        print(f"✅ [PERSONA] Saved to database: {result['pattern_name']}")
        
        return result
