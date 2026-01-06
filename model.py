from dotenv import load_dotenv
load_dotenv()  
import os
import traceback
from langchain_core.output_parsers import PydanticOutputParser
from langchain_aws import ChatBedrock
import json

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY")
LLM_MODEL_NEWS_FETCHER = os.getenv("LLM_MODEL_NEWS_FETCHER")
AWS_MODEL_REGION = os.getenv("AWS_MODEL_REGION")
llm = ChatBedrock(
    model_id=LLM_MODEL_NEWS_FETCHER,
    region_name=AWS_MODEL_REGION,   
    temperature=0.2,
    max_tokens=9999,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Async function equivalent to Vertex AI version
async def bedrock_llm(system_role: str, prompt: str, context: str):
    try:
        

        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"{prompt}\n\n\n context:{context}"}
            ],
        )
        result = response.model_dump()
        content = result["content"]
        return content
    
    except Exception as e:
        tb = traceback.format_exc()

        return f"Error: {str(e)}"