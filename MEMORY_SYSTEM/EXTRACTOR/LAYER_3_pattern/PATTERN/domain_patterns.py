from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.upsert_user_persona import upsert_user_pattern


async def detect_domain_patterns(conn, user_id: str, min_frequency: int = 2) -> list:
    """
    Detect domain patterns from problem_domain and technical_context.
    """
    
    rows = await conn.fetch("""
        SELECT 
            topic,
            COUNT(*) as frequency,
            AVG(importance_score) as avg_importance
        FROM agent_memory_system.memories
        WHERE user_id = $1
          AND status = 'active'
          AND category IN ('problem_domain', 'technical_context')
        GROUP BY topic
        HAVING COUNT(*) >= $2
        ORDER BY frequency DESC, avg_importance DESC
        LIMIT 10;
    """, user_id, min_frequency)
    
    if not rows:
        return []
    
    patterns = []
    
    total_memories = await conn.fetchval("""
        SELECT COUNT(*) 
        FROM agent_memory_system.memories
        WHERE user_id = $1 AND status = 'active';
    """, user_id)
    
    for row in rows:
        topic = row['topic']
        frequency = int(row['frequency'])
        avg_importance = float(row['avg_importance'])
        
        # Calculate confidence
        share = (frequency / total_memories) * 100 if total_memories > 0 else 0
        base_confidence = min(0.95, 0.50 + (frequency * 0.05))
        importance_boost = (avg_importance / 10) * 0.2
        confidence = min(0.95, base_confidence + importance_boost)
        
        if confidence >= 0.70:
            # 🆕 Save to database
            result = await upsert_user_pattern(
                user_id=user_id,
                pattern_type="domain",
                pattern_name=topic,
                confidence=confidence,
                frequency=frequency,
                signals_count=frequency,
                description=f"User frequently works with {topic} ({share:.1f}% of conversations)",
                metadata={"share_of_conversations": f"{share:.1f}%", "avg_importance": avg_importance}
            )
            
            patterns.append(result)
            print(f"   ✅ Domain pattern saved: {topic} (confidence: {confidence:.2f})")
    
    return patterns
