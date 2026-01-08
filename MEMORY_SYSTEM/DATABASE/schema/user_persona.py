# MEMORY_SYSTEM/database/schema/user_persona.py

from MEMORY_SYSTEM.database.connect.connect import db_manager


async def ensure_user_persona_table_exists() -> None:
    """
    Ensure user_persona table exists and matches UserPersonaModel exactly.

    Guarantees:
    - Idempotent
    - JSONB-aligned
    - Safe to run on every startup
    """

    try:
        pool = await db_manager.wait_for_connection_pool_pool()

        async with pool.acquire() as conn:
            # -------------------------------------------------
            # CORE SCHEMA & EXTENSIONS
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
            # APPLICATION SCHEMA
            # -------------------------------------------------
            await conn.execute(
                "CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;"
            )

            # -------------------------------------------------
            # USER PERSONA TABLE (JSONB ATOMIC BLOCKS)
            # -------------------------------------------------
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_memory_schema.user_persona (
                    user_id TEXT PRIMARY KEY,

                    -- Identity
                    user_identity JSONB,

                    -- Company context
                    company_profile JSONB,
                    company_business JSONB,
                    company_products JSONB,
                    company_brand JSONB,

                    -- Behavioral / content
                    objective JSONB,
                    content_format JSONB,
                    audience JSONB,
                    tone JSONB,
                    writing_style JSONB,
                    language JSONB,
                    constraints JSONB,

                    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            print("✅ user_persona table ensured successfully")

    except Exception as e:
        print(f"❌ user_persona table initialization failed: {e}")
        raise
