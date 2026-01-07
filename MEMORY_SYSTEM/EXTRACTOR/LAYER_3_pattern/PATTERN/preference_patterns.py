# MEMORY_SYSTEM/EXTRACTOR/LAYER_3_pattern/PATTERN/preference_patterns.py
# 🆕 IMPORT: Must import this at function level or top of file
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.upsert_user_persona import upsert_user_pattern
async def detect_preference_patterns(conn, user_id: str, min_frequency: int = 2) -> list:
    """
    Detect preference patterns from user_preference category memories.
    """
    
    # Query to aggregate preference topics
    rows = await conn.fetch("""
        SELECT 
            topic,
            COUNT(*) as frequency,
            AVG(importance_score) as avg_importance
        FROM agent_memory_system.memories
        WHERE user_id = $1
          AND category = 'user_preference'
          AND status = 'active'
        GROUP BY topic
        HAVING COUNT(*) >= $2
        ORDER BY frequency DESC, avg_importance DESC;
    """, user_id, min_frequency)
    
    if not rows:
        print(f"   ℹ️  No preference patterns found")
        return []
    
    patterns = []
    
    
    
    
    for row in rows:
        topic = row['topic']
        frequency = int(row['frequency'])
        avg_importance = float(row['avg_importance'])
        
        # Calculate confidence
        base_confidence = min(0.95, 0.60 + (frequency * 0.10))
        importance_boost = (avg_importance / 10) * 0.2
        confidence = min(0.95, base_confidence + importance_boost)
        
        if confidence >= 0.75:
            # 🆕 CRITICAL: Actually call upsert to save to database
            result = await upsert_user_pattern(
                user_id=user_id,
                pattern_type="preference",
                pattern_name=topic,
                confidence=confidence,
                frequency=frequency,
                signals_count=frequency,
                description=f"User has preference for {topic}",
                metadata={"avg_importance": avg_importance}
            )
            
            patterns.append(result)
            print(f"   ✅ Preference pattern saved: {topic} (confidence: {confidence:.2f}, frequency: {frequency})")
    
    return patterns
