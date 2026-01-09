# MEMORY_SYSTEM/database/schema/memory_snapshots.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_memory_snapshots_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            await conn.execute("SET search_path TO public;")
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.memory_snapshots (
                    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

                    -- ownership
                    user_id TEXT NOT NULL,

                    -- summarization
                    topic TEXT NOT NULL,
                    summary TEXT NOT NULL,

                    -- provenance
                    source_memories UUID[],

                    confidence NUMERIC(4,3) CHECK (confidence >= 0 AND confidence <= 1),
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                );
                """
            )

            print("✅ memory_snapshots table ensured successfully")

    except Exception as e:
        print(f"❌ memory_snapshots initialization failed: {e}")
        raise
