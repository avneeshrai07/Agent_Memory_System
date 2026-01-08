# MEMORY_SYSTEM/context/build_persona.py

def build_persona_context(persona: dict) -> str:
    if not persona:
        return ""

    return f"""
USER PROFILE:
- Objective: {persona.get('objective')}
- Audience: {persona.get('audience_type')}
- Tone: {persona.get('tone')}
- Style: {persona.get('style')}
- Length: {persona.get('length_preference')}
"""
