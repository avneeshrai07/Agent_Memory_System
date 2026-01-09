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

from MEMORY_SYSTEM.extraction.unified_schema import UnifiedExtractionOutput

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
    UNIFIED_EXTRACTION_SYSTEM_PROMPT = """
    You are an information extraction engine.

Your task is to extract TWO independent outputs from the conversation:

1. USER PERSONA (explicit-only, structured)
2. LONG-TERM MEMORY FACTS (explicit-only, atomic)

GLOBAL RULES (NON-NEGOTIABLE):
- Extract ONLY explicitly stated information.
- Do NOT infer, guess, or assume.
- If something is not explicitly stated, set it to null or omit it.
- One fact = one atomic statement.
- No opinions, no interpretations.

PERSONA RULES:
- Follow the persona schema strictly.
- Set fields to null if not explicitly stated.
- Confidence reflects certainty of explicit statement only.

LTM FACT RULES:
- Extract only facts worth remembering for future interactions.
- One fact per item.
- Do NOT restate persona fields unless they are factual truths.
- Facts must be historically valid statements.
- If no facts qualify, return an empty list.

Output MUST match the structured schema exactly.
No explanation text.
    """
    structured_llm = llm.with_structured_output(UnifiedExtractionOutput)
    response = await structured_llm.ainvoke(
        [
            {"role": "system", "content": UNIFIED_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    if response is None:
        return {"error": "bedrock returned null response"}

    print("agent_response_type", response)
    return response


