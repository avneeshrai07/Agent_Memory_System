# """
# Persona Extraction Prompts
# ==========================

# This file contains ALL prompt definitions used for persona extraction.
from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel
from MEMORY_SYSTEM.llm.bedrock_structured import bedrock_structured_llm_call

async def persona_extractor_function(user_prompt):

    PERSONA_EXTRACTION_SYSTEM_PROMPT = """
        You are an information extraction engine.

    Your task is to extract ONLY EXPLICITLY STATED information from the user message.
    You are strictly forbidden from guessing, inferring, assuming, or completing missing information.

    RULES (NON-NEGOTIABLE):
    1. If a field is not explicitly stated, set it to null.
    2. Do NOT infer from tone, wording, profession, or context.
    3. Do NOT deduce company details unless directly mentioned.
    4. Do NOT map implied roles, seniority, or authority.
    5. Confidence must reflect certainty of explicit statement.
    6. Never merge fields. Never generalize.
    7. If information is ambiguous, set the field to null.
    8. Output MUST be valid JSON only. No explanation text.

    If you violate these rules, the output is considered invalid.

        IMPORTANT Rules:  
    - Only extract fields that are explicitly mentioned.
    - If a field is not explicitly present, set it to null.
    - Confidence must be between 0.0 and 1.0.
    - Use confidence = 1.0 ONLY when the user states the information clearly and directly.
    - Use confidence < 1.0 ONLY if the statement is explicit but weakly asserted.
    - Do NOT create lists unless the user lists them.
    - Do NOT normalize, summarize, or enhance wording.
        """

    return await bedrock_structured_llm_call(user_prompt=user_prompt, system_prompt=PERSONA_EXTRACTION_SYSTEM_PROMPT, output_structure=UserPersonaModel)


