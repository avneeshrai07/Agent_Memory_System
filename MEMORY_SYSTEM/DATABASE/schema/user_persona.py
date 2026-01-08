#  MEMORY_SYSTEM/database/schema/user_persona.sql

from MEMORY_SYSTEM.database.connect.connect import db_manager
async def ensure_user_peronsa_table_exists():
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
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.user_persona (
                    user_id TEXT PRIMARY KEY,

                    objective TEXT,
                    desired_action TEXT,
                    success_criteria TEXT,

                    content_types TEXT[],
                    preferred_format TEXT,
                    length_preference TEXT,

                    audience_type TEXT,
                    audience_domain TEXT,
                    audience_level TEXT,

                    tone TEXT,
                    voice TEXT,
                    style TEXT,

                    language TEXT,
                    complexity TEXT,
                    jargon_policy TEXT,

                    constraints JSONB,

                    confidence REAL DEFAULT 0.5,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
            """)
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise
