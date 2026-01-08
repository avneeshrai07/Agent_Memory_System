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
    You are an AI system that infers persistent user preferences and writing expectations.

    Your task:
    - Infer preferences ONLY if they are clearly implied
    - Do NOT guess or over-infer
    - Leave fields null if uncertain
    - Confidence should reflect how strongly the signal is implied

    Rules:
    - Extract preferences, not the task itself
    - Do not restate the user's request
    - Do not include explanations
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


