"""
Persona Merger / Updater (Production-Grade)
===========================================

Purpose:
- Merge newly extracted persona data with existing persona in DB
- Apply confidence-gated overwrite rules
- Persist persona into normalized PostgreSQL schema

Design Guarantees:
- DB persona is treated as accepted truth
- Incoming persona must earn overwrite via confidence
- No lossy reconstruction of historical confidences
- No hallucinated decay logic
"""

from typing import Optional
from datetime import datetime
import traceback

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel
from MEMORY_SYSTEM.database.connect.connect import db_manager


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

CONFIDENCE_OVERRIDE_THRESHOLD = 0.80


# -------------------------------------------------------------------
# MERGE HELPERS (FIELD-LEVEL, SAFE)
# -------------------------------------------------------------------

def choose_value(
    db_value: Optional[str],
    incoming_value: Optional[str],
    incoming_confidence: float
) -> Optional[str]:
    """
    Decide whether to overwrite DB value with incoming value.
    """

    if incoming_value is None:
        return db_value

    if db_value is None:
        return incoming_value

    if incoming_confidence >= CONFIDENCE_OVERRIDE_THRESHOLD:
        return incoming_value

    return db_value


# -------------------------------------------------------------------
# MAIN MERGE + DB UPDATE (SINGLE CONNECTION)
# -------------------------------------------------------------------

async def update_user_persona(
    user_id: str,
    incoming_persona: UserPersonaModel
) -> None:
    """
    Merge incoming persona into DB persona.
    """

    print("[persona_merger] starting update for user:", user_id)

    pool = await db_manager.wait_for_connection_pool_pool()
    async with pool.acquire() as conn:

        print("[persona_merger] db connection acquired")

        row = await conn.fetchrow(
            """
            SELECT *
            FROM agentic_memory_schema.user_persona
            WHERE user_id = $1
            """,
            user_id
        )

        # Existing DB values (may be None)
        db = dict(row) if row else {}

        # -----------------------------
        # OBJECTIVE
        # -----------------------------

        objective = choose_value(
            db.get("objective"),
            incoming_persona.objective.primary_goal if incoming_persona.objective else None,
            incoming_persona.objective.confidence if incoming_persona.objective else 0.0
        )

        desired_action = choose_value(
            db.get("desired_action"),
            incoming_persona.objective.desired_action if incoming_persona.objective else None,
            incoming_persona.objective.confidence if incoming_persona.objective else 0.0
        )

        success_criteria = choose_value(
            db.get("success_criteria"),
            incoming_persona.objective.success_criteria if incoming_persona.objective else None,
            incoming_persona.objective.confidence if incoming_persona.objective else 0.0
        )

        # -----------------------------
        # CONTENT FORMAT
        # -----------------------------

        content_types = choose_value(
            db.get("content_types"),
            incoming_persona.content_format.content_types if incoming_persona.content_format else None,
            incoming_persona.content_format.confidence if incoming_persona.content_format else 0.0
        )

        preferred_format = choose_value(
            db.get("preferred_format"),
            incoming_persona.content_format.preferred_format if incoming_persona.content_format else None,
            incoming_persona.content_format.confidence if incoming_persona.content_format else 0.0
        )

        length_preference = choose_value(
            db.get("length_preference"),
            incoming_persona.content_format.length_preference if incoming_persona.content_format else None,
            incoming_persona.content_format.confidence if incoming_persona.content_format else 0.0
        )

        # -----------------------------
        # AUDIENCE
        # -----------------------------

        audience_type = choose_value(
            db.get("audience_type"),
            incoming_persona.audience.audience_type if incoming_persona.audience else None,
            incoming_persona.audience.confidence if incoming_persona.audience else 0.0
        )

        audience_domain = choose_value(
            db.get("audience_domain"),
            incoming_persona.audience.audience_domain if incoming_persona.audience else None,
            incoming_persona.audience.confidence if incoming_persona.audience else 0.0
        )

        audience_level = choose_value(
            db.get("audience_level"),
            incoming_persona.audience.audience_level if incoming_persona.audience else None,
            incoming_persona.audience.confidence if incoming_persona.audience else 0.0
        )

        # -----------------------------
        # TONE / STYLE
        # -----------------------------

        tone = choose_value(
            db.get("tone"),
            incoming_persona.tone.tone if incoming_persona.tone else None,
            incoming_persona.tone.confidence if incoming_persona.tone else 0.0
        )

        voice = choose_value(
            db.get("voice"),
            incoming_persona.tone.voice if incoming_persona.tone else None,
            incoming_persona.tone.confidence if incoming_persona.tone else 0.0
        )

        style = choose_value(
            db.get("style"),
            incoming_persona.writing_style.style if incoming_persona.writing_style else None,
            incoming_persona.writing_style.confidence if incoming_persona.writing_style else 0.0
        )

        # -----------------------------
        # LANGUAGE
        # -----------------------------

        language = choose_value(
            db.get("language"),
            incoming_persona.language.language if incoming_persona.language else None,
            incoming_persona.language.confidence if incoming_persona.language else 0.0
        )

        complexity = choose_value(
            db.get("complexity"),
            incoming_persona.language.complexity if incoming_persona.language else None,
            incoming_persona.language.confidence if incoming_persona.language else 0.0
        )

        jargon_policy = choose_value(
            db.get("jargon_policy"),
            incoming_persona.language.jargon_policy if incoming_persona.language else None,
            incoming_persona.language.confidence if incoming_persona.language else 0.0
        )

        # -----------------------------
        # CONSTRAINTS
        # -----------------------------

        constraints = choose_value(
            db.get("constraints"),
            incoming_persona.constraints.constraints if incoming_persona.constraints else None,
            incoming_persona.constraints.confidence if incoming_persona.constraints else 0.0
        )

        # -----------------------------
        # CONFIDENCE (AGGREGATED)
        # -----------------------------

        confidences = [
            p.confidence
            for p in [
                incoming_persona.objective,
                incoming_persona.content_format,
                incoming_persona.audience,
                incoming_persona.tone,
                incoming_persona.writing_style,
                incoming_persona.language,
                incoming_persona.constraints,
            ]
            if p is not None
        ]

        confidence = sum(confidences) / len(confidences) if confidences else db.get("confidence", 0.5)

        # -----------------------------
        # UPSERT
        # -----------------------------

        await conn.execute(
            """
            INSERT INTO agentic_memory_schema.user_persona (
                user_id,
                objective,
                desired_action,
                success_criteria,
                content_types,
                preferred_format,
                length_preference,
                audience_type,
                audience_domain,
                audience_level,
                tone,
                voice,
                style,
                language,
                complexity,
                jargon_policy,
                constraints,
                confidence,
                last_updated
            )
            VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                $11,$12,$13,$14,$15,$16,$17,$18,$19
            )
            ON CONFLICT (user_id)
            DO UPDATE SET
                objective = EXCLUDED.objective,
                desired_action = EXCLUDED.desired_action,
                success_criteria = EXCLUDED.success_criteria,
                content_types = EXCLUDED.content_types,
                preferred_format = EXCLUDED.preferred_format,
                length_preference = EXCLUDED.length_preference,
                audience_type = EXCLUDED.audience_type,
                audience_domain = EXCLUDED.audience_domain,
                audience_level = EXCLUDED.audience_level,
                tone = EXCLUDED.tone,
                voice = EXCLUDED.voice,
                style = EXCLUDED.style,
                language = EXCLUDED.language,
                complexity = EXCLUDED.complexity,
                jargon_policy = EXCLUDED.jargon_policy,
                constraints = EXCLUDED.constraints,
                confidence = EXCLUDED.confidence,
                last_updated = EXCLUDED.last_updated
            """,
            user_id,
            objective,
            desired_action,
            success_criteria,
            content_types,
            preferred_format,
            length_preference,
            audience_type,
            audience_domain,
            audience_level,
            tone,
            voice,
            style,
            language,
            complexity,
            jargon_policy,
            constraints,
            confidence,
            datetime.utcnow(),
        )

        print("[persona_merger] persona updated successfully")
