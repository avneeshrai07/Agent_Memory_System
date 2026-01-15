from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_artifacts_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            await conn.execute("SET search_path TO public;")
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.artifacts (
                    artifact_id UUID PRIMARY KEY,
                    artifact_type TEXT NOT NULL,

                    summary TEXT NOT NULL,
                    metadata JSONB,

                    content_ref TEXT NOT NULL,

                    created_at TIMESTAMP NOT NULL,
                    last_updated_at TIMESTAMP NOT NULL
                );
                """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_artifacts_type
                ON agentic_memory_schema.artifacts (artifact_type);
                """
            )

            print("✅ artifacts table created successfully")

    except Exception as e:
        print(f"❌ artifacts table initialization failed: {e}")
        raise
