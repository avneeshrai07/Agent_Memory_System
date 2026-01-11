"""
Persona Context Builder (JSONB-Aligned)
======================================

Purpose:
- Load user context from database
- Convert stored + provisional blocks into system-prompt shaping text
- READ-ONLY
- No LLM calls
- No DB writes

Design Rules:
- Omit unknown fields
- Never expose confidence
- Never mention internal models or storage
- Stable, reusable guidance only
"""

import json
from copy import deepcopy
from MEMORY_SYSTEM.database.connect.connect import db_manager


# -------------------------------------------------------------------
# DB LOAD
# -------------------------------------------------------------------

async def load_user_persona(user_id: str) -> dict | None:
    pool = await db_manager.wait_for_connection_pool_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM agentic_memory_schema.user_persona
            WHERE user_id = $1
            """,
            user_id,
        )

        if not row:
            return None

        data = dict(row)

        # Explicit JSONB decoding
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
            if isinstance(data.get(key), str):
                data[key] = json.loads(data[key])

        return data


# -------------------------------------------------------------------
# PROVISIONAL OVERLAY (BLOCK-AWARE)
# -------------------------------------------------------------------

def _apply_provisional_overlay(
    persona: dict,
    provisional: dict,
) -> dict:
    """
    Overlay provisional fields onto persona blocks.
    Provisional ALWAYS wins for the session.
    """

    merged = deepcopy(persona)

    FIELD_TO_BLOCK = {
        # identity
        "job_title": ("user_identity", "job_title"),
        "function": ("user_identity", "function"),
        "seniority": ("user_identity", "seniority"),
        "decision_authority": ("user_identity", "decision_authority"),

        # organization
        "company_name": ("company_profile", "company_name"),
        "industry": ("company_profile", "industry"),
        "company_size": ("company_profile", "company_size"),
        "company_stage": ("company_profile", "company_stage"),
        "business_model": ("company_business", "business_model"),
        "target_customers": ("company_business", "target_customers"),

        # objective
        "primary_goal": ("objective", "primary_goal"),
        "desired_action": ("objective", "desired_action"),
        "success_criteria": ("objective", "success_criteria"),

        # content format
        "preferred_format": ("content_format", "preferred_format"),
        "length_preference": ("content_format", "length_preference"),

        # audience
        "audience_type": ("audience", "audience_type"),
        "audience_domain": ("audience", "audience_domain"),
        "audience_level": ("audience", "audience_level"),

        # tone / style
        "tone": ("tone", "tone"),
        "voice": ("tone", "voice"),
        "emotional_intensity": ("tone", "emotional_intensity"),
        "style": ("writing_style", "style"),
        "sentence_structure": ("writing_style", "sentence_structure"),

        # language
        "language": ("language", "language"),
        "complexity": ("language", "complexity"),
        "jargon_policy": ("language", "jargon_policy"),

        # constraints
        "constraints": ("constraints", "constraints"),
    }

    for field, value in provisional.items():
        mapping = FIELD_TO_BLOCK.get(field)
        if not mapping:
            continue

        block_name, block_field = mapping

        if merged.get(block_name) is None:
            merged[block_name] = {}

        merged[block_name][block_field] = value

    return merged
# -------------------------------------------------------------------
# CONTEXT BUILDERS (BLOCK-AWARE, CONFIDENCE-FREE)
# -------------------------------------------------------------------

def _identity_context(block: dict | None) -> str | None:
    if not block:
        return None

    parts = []

    if block.get("job_title"):
        parts.append(f"The user’s role is {block['job_title']}.")

    if block.get("function"):
        parts.append(f"They work in the {block['function']} function.")

    if block.get("seniority"):
        parts.append(f"They operate at a {block['seniority'].replace('_', ' ')} level.")

    if block.get("decision_authority"):
        parts.append(
            f"They act primarily as a {block['decision_authority'].replace('_', ' ')}."
        )

    return " ".join(parts) if parts else None


def _company_context(
    profile: dict | None,
    business: dict | None,
    products: dict | None,
    brand: dict | None,
) -> str | None:
    parts = []

    if profile:
        if profile.get("company_name"):
            parts.append(f"The user is associated with {profile['company_name']}.")

        if profile.get("industry"):
            parts.append(f"The company operates in the {profile['industry']} industry.")

        if profile.get("company_size"):
            parts.append(f"The organization size is {profile['company_size']}.")

    if business:
        if business.get("business_model"):
            parts.append(f"The business follows a {business['business_model']} model.")

        if business.get("target_customers"):
            joined = ", ".join(business["target_customers"])
            parts.append(f"The primary customers are {joined}.")

    if products:
        if products.get("products"):
            names = [
                p.get("name") for p in products["products"] if p.get("name")
            ]
            if names:
                parts.append(f"Key products include: {', '.join(names)}.")

    if brand:
        if brand.get("brand_personality"):
            parts.append(
                f"The brand positioning can be described as {brand['brand_personality']}."
            )

    return " ".join(parts) if parts else None


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
            parts.append(
                f"The emotional intensity should be {block['emotional_intensity']}."
            )

    if style_block:
        if style_block.get("style"):
            parts.append(f"Adopt a {style_block['style']} writing style.")

        if style_block.get("sentence_structure"):
            parts.append(
                f"Prefer {style_block['sentence_structure']} sentences."
            )

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

async def build_persona_context(
    user_id: str,
    provisional_memory: dict | None = None,
) -> str:
    """
    Build runtime persona context.

    Priority:
    1. Provisional memory (session-level)
    2. Persisted persona (long-term)
    """

    row = await load_user_persona(user_id)
    persona = row or {}

    if provisional_memory:
        persona = _apply_provisional_overlay(persona, provisional_memory)

    sections = [
        _identity_context(persona.get("user_identity")),
        _company_context(
            persona.get("company_profile"),
            persona.get("company_business"),
            persona.get("company_products"),
            persona.get("company_brand"),
        ),
        _objective_context(persona.get("objective")),
        _format_context(persona.get("content_format")),
        _audience_context(persona.get("audience")),
        _tone_context(persona.get("tone"), persona.get("writing_style")),
        _language_context(persona.get("language")),
        _constraint_context(persona.get("constraints")),
    ]

    context_lines = [s for s in sections if s]

    if not context_lines:
        return ""

    return (
        "\n".join(f"- {line}" for line in context_lines)
    )