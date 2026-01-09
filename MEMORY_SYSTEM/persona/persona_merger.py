"""
Persona Merger / Updater (Production-Grade, JSONB-Aligned)
=========================================================

Guarantees:
- Persona blocks are atomic
- Confidence is block-scoped
- No field-level lossy merges
- Schema matches UserPersonaModel exactly
"""

import json
from datetime import datetime
from typing import Optional

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel
from MEMORY_SYSTEM.database.connect.connect import db_manager

CONFIDENCE_OVERRIDE_THRESHOLD = 0.80


# -------------------------------------------------------------------
# SAFETY HELPERS (CRITICAL)
# -------------------------------------------------------------------

def normalize_db_block(value: Optional[object]) -> Optional[dict]:
    """
    Normalize JSONB values coming from Postgres.
    asyncpg may return dict OR str depending on state.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return None


def safe_confidence(block) -> float:
    """
    Confidence must NEVER be None.
    """
    if block is None:
        return 0.0
    value = getattr(block, "confidence", 0.0)
    return float(value) if value is not None else 0.0


def jsonb_or_none(block: Optional[dict]) -> Optional[str]:
    """
    Prevent empty objects from overwriting stored data.
    """
    if not block:
        return None
    return json.dumps(block)


# -------------------------------------------------------------------
# BLOCK-LEVEL MERGE LOGIC (ATOMIC & SAFE)
# -------------------------------------------------------------------

def choose_block(
    db_block: Optional[dict],
    incoming_block: Optional[dict],
    incoming_confidence: float,
) -> Optional[dict]:
    """
    Decide whether to overwrite a persona block.

    Rules:
    - None never overwrites existing data
    - Empty dict never overwrites existing data
    - New data replaces old only if confidence ≥ threshold
    """
    if incoming_block is None:
        return db_block

    if not incoming_block:
        return db_block

    if db_block is None:
        return incoming_block

    if incoming_confidence >= CONFIDENCE_OVERRIDE_THRESHOLD:
        return incoming_block

    return db_block


# -------------------------------------------------------------------
# MAIN MERGE + DB UPSERT
# -------------------------------------------------------------------

async def update_user_persona(
    user_id: str,
    incoming_persona: UserPersonaModel,
) -> None:
    """
    Merge incoming persona into stored persona using block-level logic.
    """

    pool = await db_manager.get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM agentic_memory_schema.user_persona
            WHERE user_id = $1
            """,
            user_id,
        )

        db = dict(row) if row else {}

        # Normalize all JSONB blocks from DB
        for key in [
            "user_identity",
            "company_profile",
            "company_business",
            "company_products",
            "company_brand",
            "objective",
            "content_format",
            "audience",
            "tone",
            "writing_style",
            "language",
            "constraints",
        ]:
            db[key] = normalize_db_block(db.get(key))

        # =====================================================
        # IDENTITY
        # =====================================================
        user_identity = choose_block(
            db.get("user_identity"),
            incoming_persona.user_identity.model_dump()
            if incoming_persona.user_identity else None,
            safe_confidence(incoming_persona.user_identity),
        )

        # =====================================================
        # COMPANY CONTEXT
        # =====================================================
        company_profile = choose_block(
            db.get("company_profile"),
            incoming_persona.company_profile.model_dump()
            if incoming_persona.company_profile else None,
            safe_confidence(incoming_persona.company_profile),
        )

        company_business = choose_block(
            db.get("company_business"),
            incoming_persona.company_business.model_dump()
            if incoming_persona.company_business else None,
            safe_confidence(incoming_persona.company_business),
        )

        company_products = choose_block(
            db.get("company_products"),
            incoming_persona.company_products.model_dump()
            if incoming_persona.company_products else None,
            safe_confidence(incoming_persona.company_products),
        )

        company_brand = choose_block(
            db.get("company_brand"),
            incoming_persona.company_brand.model_dump()
            if incoming_persona.company_brand else None,
            safe_confidence(incoming_persona.company_brand),
        )

        # =====================================================
        # BEHAVIORAL / CONTENT
        # =====================================================
        objective = choose_block(
            db.get("objective"),
            incoming_persona.objective.model_dump()
            if incoming_persona.objective else None,
            safe_confidence(incoming_persona.objective),
        )

        content_format = choose_block(
            db.get("content_format"),
            incoming_persona.content_format.model_dump()
            if incoming_persona.content_format else None,
            safe_confidence(incoming_persona.content_format),
        )

        audience = choose_block(
            db.get("audience"),
            incoming_persona.audience.model_dump()
            if incoming_persona.audience else None,
            safe_confidence(incoming_persona.audience),
        )

        tone = choose_block(
            db.get("tone"),
            incoming_persona.tone.model_dump()
            if incoming_persona.tone else None,
            safe_confidence(incoming_persona.tone),
        )

        writing_style = choose_block(
            db.get("writing_style"),
            incoming_persona.writing_style.model_dump()
            if incoming_persona.writing_style else None,
            safe_confidence(incoming_persona.writing_style),
        )

        language = choose_block(
            db.get("language"),
            incoming_persona.language.model_dump()
            if incoming_persona.language else None,
            safe_confidence(incoming_persona.language),
        )

        constraints = choose_block(
            db.get("constraints"),
            incoming_persona.constraints.model_dump()
            if incoming_persona.constraints else None,
            safe_confidence(incoming_persona.constraints),
        )

        # =====================================================
        # UPSERT (JSONB SAFE)
        # =====================================================
        await conn.execute(
            """
            INSERT INTO agentic_memory_schema.user_persona (
                user_id,
                user_identity,
                company_profile,
                company_business,
                company_products,
                company_brand,
                objective,
                content_format,
                audience,
                tone,
                writing_style,
                language,
                constraints,
                last_updated
            )
            VALUES (
                $1,
                $2::jsonb,
                $3::jsonb,
                $4::jsonb,
                $5::jsonb,
                $6::jsonb,
                $7::jsonb,
                $8::jsonb,
                $9::jsonb,
                $10::jsonb,
                $11::jsonb,
                $12::jsonb,
                $13::jsonb,
                $14
            )
            ON CONFLICT (user_id)
            DO UPDATE SET
                user_identity = EXCLUDED.user_identity,
                company_profile = EXCLUDED.company_profile,
                company_business = EXCLUDED.company_business,
                company_products = EXCLUDED.company_products,
                company_brand = EXCLUDED.company_brand,
                objective = EXCLUDED.objective,
                content_format = EXCLUDED.content_format,
                audience = EXCLUDED.audience,
                tone = EXCLUDED.tone,
                writing_style = EXCLUDED.writing_style,
                language = EXCLUDED.language,
                constraints = EXCLUDED.constraints,
                last_updated = EXCLUDED.last_updated
            """,
            user_id,
            jsonb_or_none(user_identity),
            jsonb_or_none(company_profile),
            jsonb_or_none(company_business),
            jsonb_or_none(company_products),
            jsonb_or_none(company_brand),
            jsonb_or_none(objective),
            jsonb_or_none(content_format),
            jsonb_or_none(audience),
            jsonb_or_none(tone),
            jsonb_or_none(writing_style),
            jsonb_or_none(language),
            jsonb_or_none(constraints),
            datetime.utcnow(),
        )
