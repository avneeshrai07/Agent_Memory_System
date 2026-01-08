# cognition/load_cognition.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def load_cognition_config() -> dict:
    """
    Load agent cognition configuration from DB.
    Safe defaults are applied if DB fails.
    """

    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key, value
                FROM agentic_memory_schema.agent_cognition
                WHERE active = true
                """
            )

        config = {}
        for row in rows:
            config[row["key"]] = row["value"]

        return config

    except Exception:
        # Hard fallback – cognition must never block the pipeline
        return {
            "explicit_commit_threshold": 0.85,
            "implicit_confirmation_required": 2,
            "field_volatility": {},
            "volatility_penalty": {},
        }
