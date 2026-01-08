from MEMORY_SYSTEM.database.connect.connect import db_manager
import json


def _jsonify_value(value):
    """
    Ensure value is a JSON string for ::jsonb comparison.
    """
    if value is None:
        return None
    return json.dumps(value)


async def enrich_signal_frequency(
    user_id: str,
    signals: list[dict],
) -> list[dict]:
    """
    Enrich signals with historical frequency.

    Frequency definition:
    - How many times this SAME signal (category + field + value)
      has appeared before for this user.
    """

    if not signals:
        return signals

    pool = await db_manager.get_pool()

    async with pool.acquire() as conn:
        for signal in signals:
            category = signal.get("category")
            field = signal.get("field")
            value = signal.get("value")

            if not category or not field or value is None:
                signal["frequency"] = 1
                continue

            json_value = _jsonify_value(value)

            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS cnt
                FROM agentic_memory_schema.pattern_logs
                WHERE user_id = $1
                  AND signal_category = $2
                  AND signal_field = $3
                  AND signal_value = $4::jsonb
                """,
                user_id,
                category,
                field,
                json_value,
            )

            signal["frequency"] = (row["cnt"] or 0) + 1

    return signals
