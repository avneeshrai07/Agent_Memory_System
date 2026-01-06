from dotenv import load_dotenv
load_dotenv()  
import os
import traceback
from langchain_core.output_parsers import PydanticOutputParser
from langchain_aws import ChatBedrock
import json
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.EXTRACTOR.LAYER_1_llm.llm_extractor_prompt import extract_facts_from_conversation
from MEMORY_SYSTEM.EXTRACTOR.LAYER_2_Postgres.layer2_embeddings import create_embeddings
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.pattern_orchestrator import run_layer3_patterns
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
async def bedrock_llm(user_id: str, system_role: str, prompt: str, context: str):
    try:
        
        user_message = f"{prompt}\n\n\n context:{context}"

        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_message}
            ],
        )
        result = response.model_dump()
        agent_response = result["content"]
        
        
        # Parse facts safely
        facts_response = await extract_facts_from_conversation(user_message, agent_response)
        print("facts_response", facts_response)
        print("facts_response_type", type(facts_response))
        if isinstance(facts_response, str):
            print("string")
            facts = json.loads(facts_response).get("facts", [])
        elif isinstance(facts_response, dict) and "facts" in facts_response:
            print("dict")
            facts = facts_response["facts"]
        elif isinstance(facts_response, list):
            print("list")
            facts = facts_response
        else:
            print(f"Unexpected facts format: {type(facts_response)}")
            return []

        pool = await db_manager.wait_for_connection_pool_pool()
        ids = await create_embeddings(pool, user_id, facts)
        print("ids: ", ids)
        return await run_layer3_patterns(user_id)
    except Exception as e:
        tb = traceback.format_exc()

        return f"Error: {str(e)}"