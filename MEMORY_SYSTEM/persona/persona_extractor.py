"""
Persona Extractor (Structured Output)
====================================

Purpose:
- Extract user persona signals from a single interaction
- Use Bedrock structured output with Pydantic
- NO manual JSON parsing
- NO schema prompting
- NO normalization hacks

This file is intentionally simple.
"""


from dotenv import load_dotenv
load_dotenv()
from typing import Optional
import traceback
import os
from langchain_aws import ChatBedrock

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") 
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") 
AWS_MODEL_REGION = 'ap-southeast-2' 
LLM_MODEL_NEWS_FETCHER = "amazon.nova-lite-v1:0"

# -------------------------------------------------------------------
#  # BEDROCK INITIALIZATION 
# -------------------------------------------------------------------

llm = ChatBedrock( 
    model_id=LLM_MODEL_NEWS_FETCHER, 
    region_name=AWS_MODEL_REGION, 
    temperature=0.1, 
    max_tokens=9999, 
    aws_access_key_id=AWS_ACCESS_KEY_ID, 
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY 
)

# -------------------------------------------------------------------
# SYSTEM PROMPT (PERSONA INFERENCE ONLY)
# -------------------------------------------------------------------

async def persona_extractor_llm_call(
    user_prompt: str
):
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
    structured_llm = llm.with_structured_output(UserPersonaModel)
    response = await structured_llm.ainvoke(
        [
            {"role": "system", "content": PERSONA_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    if response is None:
        return {"error": "bedrock returned null response"}

    print("agent_response_type", response)
    return response


