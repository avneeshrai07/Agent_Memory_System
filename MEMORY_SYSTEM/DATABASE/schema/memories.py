from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_memories_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            # Core extensions
            await conn.execute("CREATE SCHEMA IF NOT EXISTS public;")
            await conn.execute("SET search_path TO public;")

            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;')

            # Dedicated schema
            await conn.execute("CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;")

            # DROP old table completely (user accepted data loss)
            await conn.execute("DROP TABLE IF EXISTS agentic_memory_schema.memories CASCADE;")

            # Canonical LTM table
            await conn.execute(
                """
                CREATE TABLE agentic_memory_schema.memories (
                    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                    user_id UUID NOT NULL,

                    category TEXT NOT NULL CHECK (category IN (
                        'technical_context',
                        'problem_domain',
                        'constraint',
                        'preference',
                        'expertise',
                        'validated_outcome',
                        'learned_pattern'
                    )),

                    topic TEXT NOT NULL,
                    fact TEXT NOT NULL,

                    importance REAL NOT NULL CHECK (importance >= 0 AND importance <= 10),

                    confidence_score REAL NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
                    confidence_source TEXT NOT NULL CHECK (confidence_source IN (
                        'explicit',
                        'validated',
                        'implicit'
                    )),

                    frequency INTEGER NOT NULL DEFAULT 1,

                    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
                        'active',
                        'historical',
                        'conflicting'
                    )),

                    embedding VECTOR(1024),

                    metadata JSONB NOT NULL DEFAULT '{}',

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_accessed TIMESTAMPTZ
                );
                """
            )

            # Indexes (critical)
            await conn.execute(
                "CREATE INDEX idx_memories_user ON agentic_memory_schema.memories(user_id);"
            )
            await conn.execute(
                "CREATE INDEX idx_memories_user_status ON agentic_memory_schema.memories(user_id, status);"
            )
            await conn.execute(
                "CREATE INDEX idx_memories_category ON agentic_memory_schema.memories(category);"
            )
            await conn.execute(
                "CREATE INDEX idx_memories_importance ON agentic_memory_schema.memories(importance DESC);"
            )
            await conn.execute(
                "CREATE INDEX idx_memories_embedding ON agentic_memory_schema.memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            )

            print("✅ canonical memories table created successfully")

    except Exception as e:
        print(f"❌ memories initialization failed: {e}")
        raise