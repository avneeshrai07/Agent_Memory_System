# MEMORY_SYSTEM/persona/updater.py

async def update_persona_field(
    conn,
    user_id: str,
    field: str,
    value: str,
    confidence_delta: float = 0.1
):
    await conn.execute(f"""
        INSERT INTO agentic_memory_schema.user_persona (
            user_id, {field}, confidence
        )
        VALUES ($1, $2, 0.6)
        ON CONFLICT (user_id)
        DO UPDATE SET
            {field} = EXCLUDED.{field},
            confidence = LEAST(1.0,
                agentic_memory_schema.user_persona.confidence + $3
            ),
            last_updated = NOW();
    """, user_id, value, confidence_delta)
