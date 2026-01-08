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
from datetime import datetime

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel
from MEMORY_SYSTEM.database.connect.connect import db_manager

CONFIDENCE_OVERRIDE_THRESHOLD = 0.80


# -------------------------------------------------------------------
# BLOCK-LEVEL MERGE LOGIC (CORRECT)
# -------------------------------------------------------------------

def choose_block(
    db_block: Optional[dict],
    incoming_block: Optional[dict],
    incoming_confidence: float
) -> Optional[dict]:
    """
    Decide whether to overwrite a persona block.
    """
    if incoming_block is None:
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
    incoming_persona: UserPersonaModel
) -> None:

    print("[persona_merger] starting update for user:", user_id)

    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:

        row = await conn.fetchrow(
            """
            SELECT *
            FROM agentic_memory_schema.user_persona
            WHERE user_id = $1
            """,
            user_id
        )

        db = dict(row) if row else {}

        # -----------------------------
        # BLOCK-LEVEL MERGES
        # -----------------------------

        objective = choose_block(
            db.get("objective"),
            incoming_persona.objective.model_dump() if incoming_persona.objective else None,
            incoming_persona.objective.confidence if incoming_persona.objective else 0.0
        )

        content_format = choose_block(
            db.get("content_format"),
            incoming_persona.content_format.model_dump() if incoming_persona.content_format else None,
            incoming_persona.content_format.confidence if incoming_persona.content_format else 0.0
        )

        audience = choose_block(
            db.get("audience"),
            incoming_persona.audience.model_dump() if incoming_persona.audience else None,
            incoming_persona.audience.confidence if incoming_persona.audience else 0.0
        )

        tone = choose_block(
            db.get("tone"),
            incoming_persona.tone.model_dump() if incoming_persona.tone else None,
            incoming_persona.tone.confidence if incoming_persona.tone else 0.0
        )

        writing_style = choose_block(
            db.get("writing_style"),
            incoming_persona.writing_style.model_dump() if incoming_persona.writing_style else None,
            incoming_persona.writing_style.confidence if incoming_persona.writing_style else 0.0
        )

        language = choose_block(
            db.get("language"),
            incoming_persona.language.model_dump() if incoming_persona.language else None,
            incoming_persona.language.confidence if incoming_persona.language else 0.0
        )

        constraints = choose_block(
            db.get("constraints"),
            incoming_persona.constraints.model_dump() if incoming_persona.constraints else None,
            incoming_persona.constraints.confidence if incoming_persona.constraints else 0.0
        )

        # -----------------------------
        # UPSERT (JSONB SAFE)
        # -----------------------------

        await conn.execute(
    """
    INSERT INTO agentic_memory_schema.user_persona (
        user_id,
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
        $9
    )
    ON CONFLICT (user_id)
    DO UPDATE SET
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
    json.dumps(objective) if objective else None,
    json.dumps(content_format) if content_format else None,
    json.dumps(audience) if audience else None,
    json.dumps(tone) if tone else None,
    json.dumps(writing_style) if writing_style else None,
    json.dumps(language) if language else None,
    json.dumps(constraints) if constraints else None,
    datetime.utcnow(),
)

        print("[persona_merger] persona updated successfully")
