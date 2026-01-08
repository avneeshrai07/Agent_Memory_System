#  MEMORY_SYSTEM/database/schema/pattern_logs.py

from MEMORY_SYSTEM.database.connect.connect import db_manager
async def ensure_pattern_logs_table_exists():
    try:
        pool = await db_manager.get_pool()
        async with pool.acquire() as conn:

            await conn.execute("CREATE SCHEMA IF NOT EXISTS public;")

            # ---- EXTENSIONS ----
            await conn.execute("SET search_path TO public;")

            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;')

            # ---- SCHEMAS ----

            await conn.execute('CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;')

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.pattern_logs (
                id BIGSERIAL PRIMARY KEY,

                -- ownership (MANDATORY)
                user_id TEXT NOT NULL,

                -- signal info
                signal_category TEXT NOT NULL,
                signal_field TEXT NOT NULL,

                -- cognition decision
                action TEXT NOT NULL,
                target TEXT,
                confidence NUMERIC(4,2) NOT NULL,
                reason TEXT,

                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            );
            """)
            print(f" Database initialization complete")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise
