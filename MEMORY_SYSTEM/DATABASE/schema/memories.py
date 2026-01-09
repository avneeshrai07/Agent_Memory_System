# MEMORY_SYSTEM/database/schema/memories.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_memories_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            await conn.execute("CREATE SCHEMA IF NOT EXISTS public;")
            await conn.execute("SET search_path TO public;")

            await conn.execute(
                'CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;'
            )
            await conn.execute(
                'CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;'
            )

            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.memories (
                    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

                    -- ownership (MANDATORY)
                    user_id TEXT NOT NULL,

                    -- atomic fact (IMMUTABLE MEANING)
                    fact TEXT NOT NULL,

                    -- classification
                    memory_type TEXT NOT NULL,
                    semantic_topic TEXT,

                    -- belief model
                    confidence_score NUMERIC(4,3) CHECK (confidence_score >= 0 AND confidence_score <= 1),
                    confidence_source TEXT,

                    -- reinforcement
                    evidence_count INTEGER DEFAULT 1,
                    positive_signals INTEGER DEFAULT 0,
                    negative_signals INTEGER DEFAULT 0,

                    -- lifecycle
                    status TEXT NOT NULL,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    last_seen_at TIMESTAMP WITHOUT TIME ZONE,
                    last_used_at TIMESTAMP WITHOUT TIME ZONE,

                    -- semantic search
                    embedding VECTOR(1536),

                    -- flexible metadata
                    metadata JSONB
                );
                """
            )

            print("✅ memories table ensured successfully")

    except Exception as e:
        print(f"❌ memories initialization failed: {e}")
        raise
