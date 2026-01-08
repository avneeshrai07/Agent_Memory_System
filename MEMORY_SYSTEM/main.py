

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR,  ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import asyncio
import traceback

from dotenv import load_dotenv
load_dotenv()

from MEMORY_SYSTEM.context.build_cognition_context import build_epistemic_system_prompt
from MEMORY_SYSTEM.persona.persona_agent_flow import build_user_persona_system_prompt
import traceback
from langchain_aws import ChatBedrock

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
    temperature=0.2, 
    max_tokens=9999, 
    aws_access_key_id=AWS_ACCESS_KEY_ID, 
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY 
    )

# ------------------------------------------------------------------- 
# # RAW BEDROCK CALL (NO STRUCTURED OUTPUT) 
# -------------------------------------------------------------------

async def bedrock_llm_call(
    user_id: str,
    base_system_prompt: str,
    user_prompt: str
):
    try:
        try:
            print("base_system_prompt:  ",base_system_prompt)
            epistemic_system_prompt = build_epistemic_system_prompt(base_system_prompt)
            final_system_prompt = await build_user_persona_system_prompt(user_id, epistemic_system_prompt)
            print("#"*40)
            print("final_system_prompt:  ",final_system_prompt)
            print("#"*40)
        except Exception:
            final_system_prompt = base_system_prompt

        # structured_llm = llm.with_structured_output(persona_model)

        response = await llm.ainvoke(
            [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        if response is None:
            return {"error": "bedrock returned null response"}

        agent_response = response.model_dump()
        print("agent_response:  ",agent_response)
        return agent_response['content']

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }



# -------------------------------------------------------------------
# MANUAL TEST ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    user_id="test_user_002"
    system_prmopt = "You are an assistant"
    user_prompt="""
Tell me anthing about me.
"""
    asyncio.run(
        bedrock_llm_call(
            user_id,
            system_prmopt,
            user_prompt
        )
    )
