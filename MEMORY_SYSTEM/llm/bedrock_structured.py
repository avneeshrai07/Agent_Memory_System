


from dotenv import load_dotenv
load_dotenv()
from typing import Optional
import traceback
import os
from langchain_aws import ChatBedrock
from pydantic import BaseModel


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

async def bedrock_structured_llm_call(
    system_prompt: str,
    user_prompt: str,
    output_structure: BaseModel,
    model_dump: bool = False
):
    
    structured_llm = llm.with_structured_output(output_structure)
    response = await structured_llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    if response is None:
        return {"error": "bedrock returned null response"}

    print("agent_response_type", response)
    if model_dump==True:
        extracted = response.model_dump() 
        return extracted
    else:
        return response


