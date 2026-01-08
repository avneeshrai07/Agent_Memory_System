# database/insert/log_pattern_decision.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def log_pattern_decision(
    user_id: str,
    signal: dict,
    decision: dict,
) -> None:
    """
    Persist a CognitionDecision for observability and learning.

    - User-scoped (MANDATORY)
    - Append-only
    - Must NEVER break cognition or response flow
    """

    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agentic_memory_schema.pattern_logs (
                    user_id,
                    signal_category,
                    signal_field,
                    action,
                    target,
                    confidence,
                    reason,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                """,
                user_id,
                signal.get("category"),
                signal.get("field"),
                decision.get("action"),
                decision.get("target"),
                decision.get("confidence"),
                decision.get("reason"),
            )

    except Exception:
        # Logging must NEVER break cognition or response flow
        # Fail silently or forward to app-level logger if present
        return
