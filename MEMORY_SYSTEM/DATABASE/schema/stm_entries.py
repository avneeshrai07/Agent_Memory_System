from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_stm_entries_table_exists() -> None:
    try:
        # print("[STM_INIT] Starting STM table initialization")

        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            # -------------------------------------------------
            # Core schema setup
            # -------------------------------------------------
            # print("[STM_INIT] Ensuring base schemas and extensions")

            await conn.execute("CREATE SCHEMA IF NOT EXISTS public;")
            await conn.execute("SET search_path TO public;")

            await conn.execute(
                'CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;'
            )

            # -------------------------------------------------
            # Agentic memory schema
            # -------------------------------------------------
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            # print("[STM_INIT] Creating STM entries table")

            # -------------------------------------------------
            # STM = State Memory (authoritative decisions)
            # -------------------------------------------------
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.stm_entries (
                    stm_id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,

                    state_type TEXT NOT NULL CHECK (
                        state_type IN (
                            'goal',
                            'decision',
                            'constraint',
                            'approval',
                            'rejection',
                            'direction_change',
                            'scope'
                        )
                    ),

                    statement TEXT NOT NULL,
                    rationale TEXT,

                    applies_to UUID,

                    supersedes UUID,
                    confidence FLOAT DEFAULT 1.0,

                    created_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,

                    CONSTRAINT fk_stm_supersedes
                        FOREIGN KEY (supersedes)
                        REFERENCES agentic_memory_schema.stm_entries(stm_id)
                        DEFERRABLE INITIALLY DEFERRED
                );
                """
            )

            # print("[STM_INIT] Creating STM indexes")

            # Active STM per user (hot path)
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_user_active
                ON agentic_memory_schema.stm_entries(
                    user_id,
                    is_active,
                    created_at DESC
                );
                """
            )

            # Supersession chain lookup
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_supersedes
                ON agentic_memory_schema.stm_entries(supersedes);
                """
            )

            print("✅ STM state memory table initialized successfully")

    except Exception as e:
        print(f"❌ STM initialization failed: {e}")
        raise
