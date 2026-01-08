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

                    objective JSONB,
                    content_format JSONB,
                    audience JSONB,
                    tone JSONB,
                    writing_style JSONB,
                    language JSONB,
                    constraints JSONB,

                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            print(f" Database initialization complete")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise
