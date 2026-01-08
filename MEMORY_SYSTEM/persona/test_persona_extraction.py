"""
Persona Extraction Test Runner (Bedrock)
=======================================

Purpose:
- Validate Bedrock call
- Validate persona extraction prompts
- Validate Pydantic parsing
- NO database writes
- Heavy checkpoints & defensive coding
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from dotenv import load_dotenv
load_dotenv()

import os
import json
import traceback
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Type
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import PydanticOutputParser
from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel
from MEMORY_SYSTEM.persona.persona_extractor import PersonaExtractor



# -------------------------------------------------------------------
# ENV LOADING & VALIDATION
# -------------------------------------------------------------------

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_MODEL_REGION = 'ap-southeast-2'
LLM_MODEL_NEWS_FETCHER = "amazon.nova-lite-v1:0"




# -------------------------------------------------------------------
# BEDROCK INITIALIZATION
# -------------------------------------------------------------------

llm = ChatBedrock(
    model_id=LLM_MODEL_NEWS_FETCHER,
    region_name=AWS_MODEL_REGION,   
    temperature=0.2,
    max_tokens=9999,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# -------------------------------------------------------------------
# RAW BEDROCK CALL (NO STRUCTURED OUTPUT)
# -------------------------------------------------------------------

async def bedrock_llm_call(system_prompt: str, user_prompt: str, persona_model: Type[BaseModel]):
    try:
        structured_llm = llm.with_structured_output(persona_model)

        response = await structured_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}"}
            ],
        )
        print(response)
        if response is not None:
            result = response.model_dump()
            print(result)
            return result
        else:
            return "bedrock no result"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error: {str(e)}"


# -------------------------------------------------------------------
# MAIN TEST
# -------------------------------------------------------------------

async def run_test():
    system_prompt =  "You are a professional email copywriter."

    user_prompt = """
    I need to write an email for senior executives.
    Keep it concise, professional, and focused on ROI.
    This is for a SaaS product launch.
    """

    print("INPUT user_prompt", user_prompt)

    try:
        
        persona_result = await bedrock_llm_call(system_prompt, user_prompt, UserPersonaModel)
        print("PERSONA EXTRACTOR CREATED")


        print("PERSONA EXTRACTION RESULT", persona_result)

        

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🚀 Starting Persona Extraction Test (Bedrock)\n")
    asyncio.run(run_test())
    print("\n✅ Test Finished\n")
