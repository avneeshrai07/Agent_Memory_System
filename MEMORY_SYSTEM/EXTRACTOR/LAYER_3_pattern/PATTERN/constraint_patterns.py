from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.upsert_user_persona import upsert_user_pattern

# MEMORY_SYSTEM/EXTRACTOR/LAYER_3_pattern/PATTERN/constraint_patterns.py

async def detect_constraint_patterns(conn, user_id: str, min_frequency: int = 1) -> list:
    """
    Detect constraint patterns from constraint category memories.
    """
    
    rows = await conn.fetch("""
        SELECT 
            topic,
            fact,
            COUNT(*) as frequency,
            AVG(importance_score) as avg_importance
        FROM agent_memory_system.memories
        WHERE user_id = $1
          AND category = 'constraint'
          AND status = 'active'
        GROUP BY topic, fact
        HAVING COUNT(*) >= $2
        ORDER BY avg_importance DESC;
    """, user_id, min_frequency)
    
    if not rows:
        return []
    
    patterns = []

    
    for row in rows:
        topic = row['topic']
        fact = row['fact']
        frequency = int(row['frequency'])
        
        # Constraints are always high confidence (explicit limits)
        confidence = 1.0
        
        # 🆕 Save to database
        result = await upsert_user_pattern(
            user_id=user_id,
            pattern_type="constraint",
            pattern_name=topic,
            confidence=confidence,
            frequency=frequency,
            signals_count=frequency,
            description=fact,  # Use the actual constraint fact
            metadata={"constraint_fact": fact}
        )
        
        patterns.append(result)
        print(f"   ✅ Constraint pattern saved: {topic}")
    
    return patterns
