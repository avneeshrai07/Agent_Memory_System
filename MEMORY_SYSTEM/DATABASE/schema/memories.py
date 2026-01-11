from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_memories_table_exists() -> None:
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            # -------------------------------------------------
            # Core extensions
            # -------------------------------------------------
            await conn.execute("CREATE SCHEMA IF NOT EXISTS public;")
            await conn.execute("SET search_path TO public;")

            await conn.execute(
                'CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;'
            )
            await conn.execute(
                'CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;'
            )

            # -------------------------------------------------
            # Dedicated schema
            # -------------------------------------------------
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )


            # -------------------------------------------------
            # Unified LTM table (FACTUAL + EPISODIC)
            # -------------------------------------------------
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.memories (
                    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                    user_id UUID NOT NULL,

                    -- ------------------------------------------
                    -- MEMORY DISCRIMINATOR
                    -- ------------------------------------------
                    memory_kind TEXT NOT NULL CHECK (
                        memory_kind IN ('factual', 'episodic')
                    ),

                    -- ------------------------------------------
                    -- SEMANTIC CLASSIFICATION
                    -- ------------------------------------------
                    category TEXT NOT NULL,
                    topic TEXT NOT NULL,

                    -- ------------------------------------------
                    -- CONTENT
                    -- ------------------------------------------
                    fact TEXT NOT NULL,

                    -- ------------------------------------------
                    -- SCORING / STABILITY (FACTUAL ONLY)
                    -- ------------------------------------------
                    importance REAL CHECK (importance >= 0 AND importance <= 10),

                    frequency INTEGER NOT NULL DEFAULT 1,

                    status TEXT NOT NULL DEFAULT 'active' CHECK (
                        status IN ('active', 'historical', 'conflicting')
                    ),

                    -- ------------------------------------------
                    -- CONFIDENCE
                    -- ------------------------------------------
                    confidence_score REAL NOT NULL CHECK (
                        confidence_score >= 0 AND confidence_score <= 1
                    ),

                    confidence_source TEXT NOT NULL CHECK (
                        confidence_source IN (
                            'explicit',
                            'validated',
                            'implicit',
                            'derived',
                            'inferred'
                        )
                    ),

                    -- ------------------------------------------
                    -- VECTOR (OPTIONAL FOR EPISODIC)
                    -- ------------------------------------------
                    embedding VECTOR(1024),

                    -- ------------------------------------------
                    -- EPISODIC SUPPORT
                    -- ------------------------------------------
                    expires_at TIMESTAMPTZ,

                    -- ------------------------------------------
                    -- FLEXIBLE METADATA
                    -- ------------------------------------------
                    metadata JSONB NOT NULL DEFAULT '{}',

                    -- ------------------------------------------
                    -- AUDIT
                    -- ------------------------------------------
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_accessed TIMESTAMPTZ
                );
                """
            )

            # -------------------------------------------------
            # INDEXES (CRITICAL)
            # -------------------------------------------------

            # User + kind (most common filter)
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_kind
                ON agentic_memory_schema.memories(user_id, memory_kind);
                """
            )

            # Episodic expiry scan
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_episodic_expiry
                ON agentic_memory_schema.memories(expires_at)
                WHERE memory_kind = 'episodic';
                """
            )

            # Factual retrieval
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_factual_confidence
                ON agentic_memory_schema.memories(confidence_score DESC)
                WHERE memory_kind = 'factual';
                """
            )

            # Vector search (FACTUAL ONLY)
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON agentic_memory_schema.memories
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )

            print("✅ unified factual + episodic memories table created successfully")

    except Exception as e:
        print(f"❌ memories initialization failed: {e}")
        raise
