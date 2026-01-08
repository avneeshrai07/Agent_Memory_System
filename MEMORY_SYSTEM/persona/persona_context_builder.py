"""
Persona Context Builder (JSONB-Aligned)
======================================

Purpose:
- Load user persona from database
- Convert persona blocks into system-prompt shaping text
- READ-ONLY
- No LLM calls
- No DB writes

Design Rules:
- Omit unknown fields
- Never expose confidence
- Never mention 'persona' or 'profile'
- Stable, reusable guidance only
"""

from MEMORY_SYSTEM.database.connect.connect import db_manager


# -------------------------------------------------------------------
# DB LOAD
# -------------------------------------------------------------------

import json

async def load_user_persona(user_id: str) -> dict | None:
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

        if not row:
            return None

        data = dict(row)

        # Explicit JSONB decoding
        for key in [
            "objective",
            "content_format",
            "audience",
            "tone",
            "writing_style",
            "language",
            "constraints",
        ]:
            if isinstance(data.get(key), str):
                data[key] = json.loads(data[key])

        return data


# -------------------------------------------------------------------
# CONTEXT BUILDERS (JSONB BLOCK-AWARE)
# -------------------------------------------------------------------

def _objective_context(block: dict | None) -> str | None:
    if not block:
        return None

    parts = []

    if block.get("primary_goal"):
        parts.append(f"The primary goal of the content is to {block['primary_goal']}.")

    if block.get("desired_action"):
        parts.append(f"The desired reader action is: {block['desired_action']}.")

    if block.get("success_criteria"):
        parts.append(f"Success is defined as: {block['success_criteria']}.")

    return " ".join(parts) if parts else None


def _format_context(block: dict | None) -> str | None:
    if not block:
        return None

    parts = []

    if block.get("content_types"):
        parts.append(
            f"The content is typically delivered as: {', '.join(block['content_types'])}."
        )

    if block.get("preferred_format"):
        parts.append(f"Use a {block['preferred_format']} format.")

    if block.get("length_preference"):
        parts.append(f"Keep the length {block['length_preference']}.")

    return " ".join(parts) if parts else None


def _audience_context(block: dict | None) -> str | None:
    if not block:
        return None

    parts = []

    if block.get("audience_type"):
        parts.append(f"The target audience is {block['audience_type']}.")

    if block.get("audience_domain"):
        parts.append(f"The audience operates in the {block['audience_domain']} domain.")

    if block.get("audience_level"):
        parts.append(f"The audience knowledge level is {block['audience_level']}.")

    return " ".join(parts) if parts else None


def _tone_context(block: dict | None, style_block: dict | None) -> str | None:
    parts = []

    if block:
        if block.get("tone"):
            parts.append(f"Maintain a {block['tone']} tone.")

        if block.get("voice"):
            parts.append(f"Use {block['voice'].replace('_', ' ')} voice.")

        if block.get("emotional_intensity"):
            parts.append(f"The emotional intensity should be {block['emotional_intensity']}.")

    if style_block:
        if style_block.get("style"):
            parts.append(f"Adopt a {style_block['style']} writing style.")

        if style_block.get("sentence_structure"):
            parts.append(f"Prefer {style_block['sentence_structure']} sentences.")

    return " ".join(parts) if parts else None


def _language_context(block: dict | None) -> str | None:
    if not block:
        return None

    parts = []

    if block.get("language"):
        parts.append(f"Write in {block['language']}.")

    if block.get("complexity"):
        parts.append(f"Use a {block['complexity']} level of complexity.")

    if block.get("jargon_policy"):
        if block["jargon_policy"] == "avoid":
            parts.append("Avoid jargon.")
        elif block["jargon_policy"] == "required":
            parts.append("Use domain-specific terminology where appropriate.")
        else:
            parts.append("Jargon is acceptable where helpful.")

    return " ".join(parts) if parts else None


def _constraint_context(block: dict | None) -> str | None:
    if not block:
        return None

    constraints = block.get("constraints")
    if constraints:
        joined = "; ".join(constraints)
        return f"Respect the following constraints: {joined}."

    return None


# -------------------------------------------------------------------
# PUBLIC ENTRY POINT
# -------------------------------------------------------------------

async def build_persona_context(user_id: str) -> str:
    row = await load_user_persona(user_id)

    if not row:
        return ""
    print("database_row:    ", row)
    sections = [
        _objective_context(row.get("objective")),
        _format_context(row.get("content_format")),
        _audience_context(row.get("audience")),
        _tone_context(row.get("tone"), row.get("writing_style")),
        _language_context(row.get("language")),
        _constraint_context(row.get("constraints")),
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
