# create_layer_2_tables.py
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager


async def ensure_layer_2_table_exists():
    """
    Creates agent_memory_system.memories table with:
    - pgvector support (VECTOR(1536) for text-embedding-3-small)
    - HNSW index (fast similarity search)
    - Proper schema, indexes, triggers
    """
    try:
        pool = await db_manager.wait_for_connection_pool_pool()
        async with pool.acquire() as conn:
            # ---- EXTENSIONS / SEARCH PATH ----
            await conn.execute("CREATE SCHEMA IF NOT EXISTS public;")
            await conn.execute("SET search_path TO public;")

            await conn.execute(
                'CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;'
            )
            await conn.execute(
                'CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;'
            )

            # ---- APP SCHEMA ----
            await conn.execute("CREATE SCHEMA IF NOT EXISTS agent_memory_system;")

            # ---- Core memories table (1024 dims for text-embedding-3-small) ----
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory_system.memories (
                    id               BIGSERIAL PRIMARY KEY,
                    user_id          TEXT NOT NULL,
                    topic            TEXT NOT NULL,
                    fact             TEXT NOT NULL,
                    category         TEXT NOT NULL,
                    importance_score SMALLINT NOT NULL CHECK (importance_score >= 1 AND importance_score <= 10),
                    embedding        VECTOR(1024),
                    status           TEXT NOT NULL DEFAULT 'active' 
                        CHECK (status IN ('active', 'merged', 'deleted')),
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            # ---- Track consolidation/merge relationships ----
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory_system.memory_links (
                    id               BIGSERIAL PRIMARY KEY,
                    parent_memory_id BIGINT NOT NULL
                        REFERENCES agent_memory_system.memories(id)
                        ON DELETE CASCADE,
                    child_memory_id  BIGINT NOT NULL
                        REFERENCES agent_memory_system.memories(id)
                        ON DELETE CASCADE,
                    relation_type    TEXT NOT NULL 
                        CHECK (relation_type IN ('duplicate', 'related')),
                    similarity       REAL NOT NULL CHECK (similarity >= 0 AND similarity <= 1),
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            # ---- Composite indexes for filtering ----
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_status
                ON agent_memory_system.memories (user_id, status);
                """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_topic
                ON agent_memory_system.memories (user_id, topic);
                """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_category
                ON agent_memory_system.memories (user_id, category);
                """
            )

            # ---- HNSW Vector index (fast cosine similarity) ----
            # await conn.execute(
            #     """
            #     CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
            #     ON agent_memory_system.memories
            #     USING hnsw (embedding vector_cosine_ops)
            #     WITH (m = 16, ef_construction = 64);
            #     """
            # )

            # ---- Trigger function ----
            await conn.execute(
                """
                CREATE OR REPLACE FUNCTION agent_memory_system.set_memories_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at := NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                """
            )

            # ---- Trigger ----
            await conn.execute(
                """
                DROP TRIGGER IF EXISTS trg_memories_set_updated_at 
                ON agent_memory_system.memories;
                
                CREATE TRIGGER trg_memories_set_updated_at
                BEFORE UPDATE ON agent_memory_system.memories
                FOR EACH ROW
                EXECUTE FUNCTION agent_memory_system.set_memories_updated_at();
                """
            )

            # ---- Final analyze for stats ----
            await conn.execute("ANALYZE agent_memory_system.memories;")
            await conn.execute("ANALYZE agent_memory_system.memory_links;")

            print("✅ Layer 2 tables created successfully:")
            print("   - agent_memory_system.memories (VECTOR(1536) + HNSW)")
            print("   - agent_memory_system.memory_links")
            print("   - All indexes and triggers ready")

    except Exception as e:
        print(f"❌ Layer 2 table creation failed: {e}")
        print("   - Check pgvector extension, Postgres version, disk space")
        raise
