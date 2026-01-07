# create_layer_2_tables.py
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager


async def ensure_layer_2_table_exists():
    """
    Creates agent_memory_system.memories table with:
    - pgvector support (VECTOR(1024) for AWS Bedrock embeddings)
    - HNSW index (fast cosine similarity search)
    - Consolidation support (frequency, merged_into_id, needs_consolidation)
    - Proper schema, indexes, and triggers
    """
    try:
        pool = await db_manager.wait_for_connection_pool_pool()
        async with pool.acquire() as conn:
            
            # ================================================================
            # EXTENSIONS & SCHEMA
            # ================================================================
            await conn.execute("CREATE SCHEMA IF NOT EXISTS public;")
            await conn.execute("SET search_path TO public;")
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;')
            await conn.execute("CREATE SCHEMA IF NOT EXISTS agent_memory_system;")

            # ================================================================
            # TABLE: memories (core LTM storage with consolidation support)
            # ================================================================
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory_system.memories (
                    -- Identity
                    id                  BIGSERIAL PRIMARY KEY,
                    user_id             TEXT NOT NULL,
                    
                    -- Extracted fact content
                    topic               TEXT NOT NULL,
                    fact                TEXT NOT NULL,
                    category            TEXT NOT NULL 
                        CHECK (category IN (
                            'technical_context',
                            'user_preference', 
                            'problem_domain',
                            'expertise',
                            'constraint',
                            'learned_pattern'
                        )),
                    importance_score    SMALLINT NOT NULL 
                        CHECK (importance_score >= 1 AND importance_score <= 10),
                    
                    -- Vector embedding (1024 dims for Bedrock)
                    embedding           VECTOR(1024),
                    
                    -- Consolidation & lifecycle fields
                    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'merged', 'deleted', 'historical', 'conflicting')),
                    frequency           INT NOT NULL DEFAULT 1,
                    merged_into_id      BIGINT NULL 
                        REFERENCES agent_memory_system.memories(id) ON DELETE SET NULL,
                    needs_consolidation BOOLEAN NOT NULL DEFAULT TRUE,
                    
                    -- Retrieval tracking (for importance scoring)
                    last_accessed_at    TIMESTAMPTZ NULL,
                    access_count        INT NOT NULL DEFAULT 0,
                    
                    -- Metadata (store source message IDs, extractor version, etc.)
                    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
                    
                    -- Timestamps
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # ================================================================
            # TABLE: memory_links (merge/similarity relationships)
            # ================================================================
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory_system.memory_links (
                    id                  BIGSERIAL PRIMARY KEY,
                    parent_memory_id    BIGINT NOT NULL
                        REFERENCES agent_memory_system.memories(id) ON DELETE CASCADE,
                    child_memory_id     BIGINT NOT NULL
                        REFERENCES agent_memory_system.memories(id) ON DELETE CASCADE,
                    relation_type       TEXT NOT NULL 
                        CHECK (relation_type IN ('duplicate', 'related', 'contradicts')),
                    similarity          REAL NOT NULL 
                        CHECK (similarity >= 0 AND similarity <= 1),
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    -- Prevent duplicate links
                    CONSTRAINT uq_memory_links_edge 
                        UNIQUE (parent_memory_id, child_memory_id, relation_type)
                );
            """)

            # ================================================================
            # INDEXES: Filtering, consolidation queue, and vector search
            # ================================================================
            
            # User + status (primary retrieval filter)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_status
                ON agent_memory_system.memories (user_id, status);
            """)

            # Active memories only (partial index for faster scans)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_active
                ON agent_memory_system.memories (user_id, created_at DESC)
                WHERE status = 'active';
            """)

            # Consolidation queue (new unconsolidated memories)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_consolidation_queue
                ON agent_memory_system.memories (user_id, created_at)
                WHERE needs_consolidation = true AND status = 'active';
            """)

            # Topic aggregation (for domain pattern detection)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_topic
                ON agent_memory_system.memories (user_id, topic)
                WHERE status = 'active';
            """)

            # Category aggregation (for preference/constraint detection)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_category
                ON agent_memory_system.memories (user_id, category)
                WHERE status = 'active';
            """)

            # Vector similarity search (HNSW for cosine distance)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
                ON agent_memory_system.memories
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)

            # Memory links indexes (find children/parents efficiently)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_links_parent
                ON agent_memory_system.memory_links (parent_memory_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_links_child
                ON agent_memory_system.memory_links (child_memory_id);
            """)

            # ================================================================
            # TRIGGER: Auto-update updated_at on row modification
            # ================================================================
            await conn.execute("""
                CREATE OR REPLACE FUNCTION agent_memory_system.set_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at := NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            await conn.execute("""
                DROP TRIGGER IF EXISTS trg_memories_set_updated_at 
                ON agent_memory_system.memories;
                
                CREATE TRIGGER trg_memories_set_updated_at
                BEFORE UPDATE ON agent_memory_system.memories
                FOR EACH ROW
                EXECUTE FUNCTION agent_memory_system.set_updated_at();
            """)

            # ================================================================
            # ANALYZE: Refresh query planner statistics
            # ================================================================
            await conn.execute("ANALYZE agent_memory_system.memories;")
            await conn.execute("ANALYZE agent_memory_system.memory_links;")

            # ================================================================
            # SUCCESS
            # ================================================================
            print("✅ Layer 2 tables created successfully:")
            print("   📊 agent_memory_system.memories")
            print("      - VECTOR(1024) with HNSW index (cosine similarity)")
            print("      - Consolidation columns: frequency, merged_into_id, needs_consolidation")
            print("      - Status tracking: active, merged, deleted, historical, conflicting")
            print("   🔗 agent_memory_system.memory_links")
            print("      - Relation types: duplicate, related, contradicts")
            print("   🚀 All indexes, constraints, and triggers ready")

    except Exception as e:
        print(f"❌ Layer 2 table creation failed: {e}")
        print("   - Check: pgvector extension installed?")
        print("   - Check: Postgres version >= 12?")
        print("   - Check: Disk space available?")
        raise
