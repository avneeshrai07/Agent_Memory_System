async def detect_persona_shift(conn, user_id: str):
    """
    Detect if user's tone/style is shifting.
    If new signals contradict current persona, mark old as historical.
    """
    
    # Get current persona
    current_persona = await conn.fetchrow("""
        SELECT pattern_name, metadata
        FROM agent_memory_system.user_patterns
        WHERE user_id = $1 
          AND pattern_type = 'persona'
          AND status = 'active'
        ORDER BY last_observed DESC
        LIMIT 1;
    """, user_id)
    
    if not current_persona:
        return  # No persona yet
    
    current_tone = current_persona['metadata'].get('tone')
    
    # Check for contradicting tone signals (last 7 days)
    contradicting_signals = await conn.fetch("""
        SELECT 
            pattern_name,
            COUNT(*) as frequency
        FROM agent_memory_system.user_patterns
        WHERE user_id = $1
          AND pattern_type = 'preference'
          AND pattern_name ILIKE '%tone%'
          AND pattern_name NOT ILIKE $2
          AND last_observed > NOW() - INTERVAL '7 days'
          AND status = 'active'
        GROUP BY pattern_name
        HAVING COUNT(*) >= 3  -- 3+ signals = shift
        ORDER BY frequency DESC;
    """, user_id, f'%{current_tone}%')
    
    if contradicting_signals:
        # Mark old persona as historical
        await conn.execute("""
            UPDATE agent_memory_system.user_patterns
            SET status = 'historical'
            WHERE user_id = $1 
              AND pattern_type = 'persona'
              AND status = 'active';
        """, user_id)
        
        print(f"🔄 Persona shift detected for {user_id}: {current_tone} → {contradicting_signals[0]['pattern_name']}")
