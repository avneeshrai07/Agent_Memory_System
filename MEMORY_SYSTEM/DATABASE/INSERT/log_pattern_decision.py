import json
from MEMORY_SYSTEM.database.connect.connect import db_manager


async def log_pattern_decision(
    user_id: str,
    signal: dict,
    decision: dict,
) -> None:
    """
    Persist a CognitionDecision for observability and learning.

    Guarantees:
    - User-scoped
    - Append-only
    - Never blocks cognition
    """

    try:
        pool = await db_manager.get_pool()

        # ---- FIX 1: JSON-safe signal value ----
        signal_value = signal.get("value")
        if signal_value is not None:
            signal_value = json.dumps(signal.get("value"))

        # ---- FIX 2: confidence hardening ----
        confidence = decision.get("confidence")
        if confidence is None:
            confidence = 0.0
        confidence = round(float(confidence), 2)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agentic_memory_schema.pattern_logs (
                    user_id,
                    signal_category,
                    signal_field,
                    signal_value,
                    action,
                    target,
                    confidence,
                    reason,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                """,
                user_id,
                signal.get("category"),
                signal.get("field"),
                signal_value,
                decision.get("action"),
                decision.get("target"),
                confidence,
                decision.get("reason"),
            )

    except Exception as e:
        print("❌ pattern_logs insert failed")
        print("signal =", signal)
        print("decision =", decision)
        print("error =", e)
        return
