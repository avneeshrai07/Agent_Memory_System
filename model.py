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
        

        print(f"📝 [SMART CONTEXT]\n{smart_context}\n")

        # User message (DON'T dump raw patterns, use smart context instead)
        user_message = f"{prompt}\n\nsome contex about user {smart_context}  \n\n Additional context: {context}"
        
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
