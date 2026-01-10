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
                memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                user_id TEXT NOT NULL,

                fact TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                semantic_topic TEXT,

                confidence_score REAL NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
                confidence_source TEXT NOT NULL DEFAULT 'explicit',

                status TEXT NOT NULL DEFAULT 'active',

                embedding VECTOR(1024) NOT NULL,

                evidence_count INTEGER NOT NULL DEFAULT 1,

                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
                """
            )

            print("✅ memories table ensured successfully")

    except Exception as e:
        print(f"❌ memories initialization failed: {e}")
        raise
