"""
Persona Extraction Prompts
==========================

This file contains ALL prompt definitions used for persona extraction.

Rules:
- Prompts must be conservative and non-imaginative
- No defaults, no guessing
- Absence of evidence = null
- Prompts must align 1:1 with Pydantic schemas
"""

BASE_SYSTEM_PROMPT = """
You are a persona extraction engine for a content generation platform.

Your task is to extract durable user persona attributes from a conversation.

STRICT RULES:
- Extract only what is explicitly stated or strongly implied through repeated behavior.
- Never guess, assume, or fill missing information.
- If a field is not clearly supported, return null.
- Do NOT optimize for completeness.
- Persona attributes must be stable across sessions.
- Output must strictly conform to the provided JSON schema.
- Do not include explanations, comments, or additional text.
"""


def build_persona_prompt(conversation: str, schema_json: str) -> str:
    """
    Builds a persona extraction prompt tied strictly to a schema.
    """
    return f"""
Analyze the conversation below and extract ONLY the persona attributes
that are clearly supported by evidence.

Conversation:
{conversation}

Return JSON that strictly matches this schema:
{schema_json}
"""
