# MEMORY_SYSTEM/database/schema/memory_access_log.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_memory_access_log_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            await conn.execute("SET search_path TO public;")
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.memory_access_log (
                    access_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

                    memory_id UUID NOT NULL
                        REFERENCES agentic_memory_schema.memories(memory_id)
                        ON DELETE CASCADE,

                    accessed_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    access_reason TEXT,
                    relevance_score NUMERIC(6,4)
                );
                """
            )

            print("✅ memory_access_log table ensured successfully")

    except Exception as e:
        print(f"❌ memory_access_log initialization failed: {e}")
        raise
