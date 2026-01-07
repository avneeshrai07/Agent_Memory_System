from dotenv import load_dotenv
load_dotenv()  
import os
import traceback
from langchain_core.output_parsers import PydanticOutputParser
from langchain_aws import ChatBedrock
import json
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
# from MEMORY_SYSTEM.EXTRACTOR.LAYER_1_llm.llm_extractor_prompt import extract_facts_from_conversation
# from MEMORY_SYSTEM.EXTRACTOR.LAYER_2_Postgres.layer2_embeddings import create_embeddings
# from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.detect_patterns import detect_all_patterns
# from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.PATTERN.persona_detector import detect_user_persona
from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.load_patterns import load_all_user_patterns
from MEMORY_SYSTEM.main import process_conversation_memory

from MEMORY_SYSTEM.main import build_context_with_persona
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
# Async function equivalent to Vertex AI version
# In your model.py or wherever bedrock_llm is defined

from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.context_builder import build_smart_context, build_compact_context


async def bedrock_llm(user_id: str, system_role: str, prompt: str, context: str):
    try:
        # ====================================================================
        # STEP 1: Load persona and build adapted system prompt
        # ====================================================================
        persona = await build_context_with_persona(user_id, prompt)
        
        
        # ====================================================================
        # STEP 2: Load ALL patterns and build smart context
        # ====================================================================
        user_patterns = await load_all_user_patterns(user_id)
        
        # 🆕 BUILD SMART CONTEXT (choose one based on your needs)
        if user_patterns['persona'] or user_patterns['preferences'] or user_patterns['domains']:
            smart_context = build_smart_context(user_patterns)  # Detailed
            # OR
            # smart_context = build_compact_context(user_patterns)  # Compact
        else:
            smart_context = "No user context available yet."
        
        # ====================================================================
        # STEP 3: Combine into system prompt
        # ====================================================================
        


        # User message (DON'T dump raw patterns, use smart context instead)
        user_message = f"{prompt}\n\nsome contex about user {smart_context}  \n\n Additional context: {context}"
        
        print(f"📝 [SMART CONTEXT]\n{smart_context}\n")
        print(f"user_message: {user_message}")
        
        # ====================================================================
        # STEP 4: Call LLM with enhanced context
        # ====================================================================
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_message}
            ],
        )
        result = response.model_dump()
        agent_response = result["content"]
        
        print("✅ Agent response generated")
        
        # ====================================================================
        # STEP 5: Process memory in background
        # ====================================================================
        import asyncio
        asyncio.create_task(
            process_conversation_memory(
                user_id=user_id,
                user_message=prompt,
                agent_response=agent_response
            )
        )
        
        print("🚀 Memory processing started in background")
        
        return agent_response
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ Error in bedrock_llm: {tb}")
        return f"Error: {str(e)}"
