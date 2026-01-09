# MEMORY_SYSTEM/database/schema/memory_links.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_memory_links_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            await conn.execute("SET search_path TO public;")
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.memory_links (
                    link_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

                    from_memory_id UUID NOT NULL
                        REFERENCES agentic_memory_schema.memories(memory_id)
                        ON DELETE CASCADE,

                    to_memory_id UUID NOT NULL
                        REFERENCES agentic_memory_schema.memories(memory_id)
                        ON DELETE CASCADE,

                    relation_type TEXT NOT NULL,
                    confidence NUMERIC(4,3) CHECK (confidence >= 0 AND confidence <= 1),

                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                );
                """
            )

            print("✅ memory_links table ensured successfully")

    except Exception as e:
        print(f"❌ memory_links initialization failed: {e}")
        raise
