"""
Persona Context Builder
=======================

Purpose:
- Load user persona from database
- Convert persona fields into system-prompt shaping text
- READ-ONLY
- No LLM calls
- No DB writes

Design Rules:
- Omit unknown (NULL) fields
- Never expose confidence to the model
- Never mention 'persona' or 'profile'
- Produce stable, reusable guidance
"""

from MEMORY_SYSTEM.database.connect.connect import db_manager


# -------------------------------------------------------------------
# DB LOAD
# -------------------------------------------------------------------

async def load_user_persona(user_id: str) -> dict | None:
    """
    Load persona row from DB.
    """

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

        return dict(row) if row else None


# -------------------------------------------------------------------
# CONTEXT BUILDERS (SECTIONED, STABLE)
# -------------------------------------------------------------------

def _objective_context(row: dict) -> str | None:
    parts = []

    if row.get("objective"):
        parts.append(f"The primary goal of the content is to {row['objective']}.")

    if row.get("desired_action"):
        parts.append(f"The desired reader action is: {row['desired_action']}.")

    if row.get("success_criteria"):
        parts.append(f"Success is defined as: {row['success_criteria']}.")

    return " ".join(parts) if parts else None


def _format_context(row: dict) -> str | None:
    parts = []

    if row.get("content_types"):
        parts.append(
            f"The user typically requests content such as: {', '.join(row['content_types'])}."
        )

    if row.get("preferred_format"):
        parts.append(f"Use a {row['preferred_format']} format.")

    if row.get("length_preference"):
        parts.append(f"Keep the length {row['length_preference']}.")

    return " ".join(parts) if parts else None


def _audience_context(row: dict) -> str | None:
    parts = []

    if row.get("audience_type"):
        parts.append(f"The target audience is {row['audience_type']}.")

    if row.get("audience_domain"):
        parts.append(f"The audience operates in the {row['audience_domain']} domain.")

    if row.get("audience_level"):
        parts.append(f"The audience knowledge level is {row['audience_level']}.")

    return " ".join(parts) if parts else None


def _tone_context(row: dict) -> str | None:
    parts = []

    if row.get("tone"):
        parts.append(f"Maintain a {row['tone']} tone.")

    if row.get("voice"):
        parts.append(f"Use {row['voice'].replace('_', ' ')} voice.")

    if row.get("style"):
        parts.append(f"Adopt a {row['style']} writing style.")

    return " ".join(parts) if parts else None


def _language_context(row: dict) -> str | None:
    parts = []

    if row.get("language"):
        parts.append(f"Write in {row['language']}.")

    if row.get("complexity"):
        parts.append(f"Use a {row['complexity']} level of complexity.")

    if row.get("jargon_policy"):
        if row["jargon_policy"] == "avoid":
            parts.append("Avoid jargon.")
        elif row["jargon_policy"] == "required":
            parts.append("Use domain-specific terminology where appropriate.")
        else:
            parts.append("Jargon is acceptable where helpful.")

    return " ".join(parts) if parts else None


def _constraint_context(row: dict) -> str | None:
    if row.get("constraints"):
        joined = "; ".join(row["constraints"])
        return f"Respect the following constraints: {joined}."
    return None


# -------------------------------------------------------------------
# PUBLIC ENTRY POINT
# -------------------------------------------------------------------

async def build_persona_context(user_id: str) -> str:
    """
    Build system-prompt persona context for a user.
    """

    row = await load_user_persona(user_id)

    if not row:
        return ""

    sections = [
        _objective_context(row),
        _format_context(row),
        _audience_context(row),
        _tone_context(row),
        _language_context(row),
        _constraint_context(row),
    ]

    context_lines = [s for s in sections if s]

    if not context_lines:
        return ""

    final_context = (
        "USER CONTEXT:\n"
        + "\n".join(f"- {line}" for line in context_lines)
    )

    print("[persona_context_builder] persona context built")

    return final_context
