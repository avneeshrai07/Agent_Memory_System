

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
from fastapi import BackgroundTasks
from MEMORY_SYSTEM.runtime.background_worker import submit_background_task
from MEMORY_SYSTEM.context.build_cognition_context import build_epistemic_system_prompt
from MEMORY_SYSTEM.persona.persona_agent_flow import bring_user_persona
from MEMORY_SYSTEM.persona.persona_agent_flow import learn_persona_from_interaction
from MEMORY_SYSTEM.ltm.extract_ltm import extract_ltm_facts
from MEMORY_SYSTEM.ltm.retriever import retrieve_ltm_memories
from MEMORY_SYSTEM.ltm.context_builder import build_ltm_context
from MEMORY_SYSTEM.stm.stm_orchestrator import process_user_message, post_model_response

import traceback
from langchain_aws import ChatBedrock

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") 
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") 
AWS_MODEL_REGION = 'ap-southeast-2' 
LLM_MODEL_NEWS_FETCHER = "amazon.nova-lite-v1:0"


import asyncio
from typing import Set

BACKGROUND_TASKS: Set[asyncio.Task] = set()


def register_task(task: asyncio.Task) -> None:
    BACKGROUND_TASKS.add(task)

    def _cleanup(t: asyncio.Task):
        BACKGROUND_TASKS.discard(t)

    task.add_done_callback(_cleanup)

def safe_bg(task_name: str, factory):
    def _wrapped():
        # This returns the coroutine directly, not awaiting it
        return factory()  # Problem: worker expects this to be a coroutine
    return _wrapped


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
    system_prompt: str,
    user_prompt: str,
    background_tasks: BackgroundTasks
):
    try:
        print("\n\n===================SYSTEM PROMPT START===================\n\n")
        print(system_prompt)
        print("\n\n===================SYSTEM PROMPT END===================\n\n")

        print("\n\n===================USER PROMPT START===================\n\n")
        print(user_prompt)
        print("\n\n===================USER PROMPT END===================\n\n")

        user_intent = await process_user_message(
        user_id,
        user_prompt
    )
        print("user_intent:  ", user_intent)
        try:
            epistemic_system_prompt = build_epistemic_system_prompt(system_prompt)
            print("\n\n===================EPISTEMIC SYSTEM PROMPT START===================\n\n")
            print(epistemic_system_prompt)
            print("\n\n===================EPISTEMIC SYSTEM PROMPT END===================\n\n")
            user_persona = await bring_user_persona(user_id)
            print("\n\n===================USER PERSONA START===================\n\n")
            print(user_persona)
            print("\n\n===================USER PERSONA END===================\n\n")

            final_system_prompt = f"""
            RULES:
            {epistemic_system_prompt}

            USER_PERSONA:
            {user_persona}
"""
        except Exception:
            final_system_prompt = system_prompt


        try:
            ltm_memories = await retrieve_ltm_memories(user_id, user_prompt)
            episodic = ltm_memories.get("episodic",None)
            factual = ltm_memories.get("factual",None)
            ltm_context = build_ltm_context(factual)
            print("\n\n===================LONG TERMS EPISODIC MEMORIES START===================\n\n")
            print(episodic)
            print("\n\n===================LONG TERM EPISODIC MEMORIES END===================\n\n")

            print("\n\n===================LONG TERMS FACTUAL MEMORIES START===================\n\n")
            print(factual)
            print("\n\n===================LONG TERM FACTUAL MEMORIES END===================\n\n")

            final_user_prompt = f"""
            {ltm_context}

            User question:
            {user_prompt}
            """
        except Exception:
            final_user_prompt = user_prompt


        print("#"*40)
        print("final_system_prompt:  ",final_system_prompt)
        print("*"*40)
        print("final_user_prompt:  ",final_user_prompt)
        print("#"*40)


        # structured_llm = llm.with_structured_output(persona_model)



        response = await llm.ainvoke(
            [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": final_user_prompt}
            ]
        )
        # --------------------------------------------------
        # BACKGROUND PERSONA LEARNING (NON-BLOCKING)
        # --------------------------------------------------
        # If you await a function, it must NOT be submitted to background
        

        if response is None:
            return {"error": "bedrock returned null response"}

        agent_response = response.model_dump()
        # print("agent_response:  ",agent_response)
        print(agent_response.get('content'))
        agent_response_content = agent_response.get('content')

        background_tasks.add_task(
            post_model_response,
            user_id=user_id,
            route=user_intent["route"],
            route_confidence=user_intent["route_confidence"],
            stm_written=user_intent["stm_written"],
            response_text=agent_response_content
        )
        
        background_tasks.add_task(
            learn_persona_from_interaction,
            user_id,
            user_prompt
        )
        
        background_tasks.add_task(
            extract_ltm_facts,
            user_id,
            user_prompt,
            agent_response_content
        )

        return agent_response_content

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }



# -------------------------------------------------------------------
# MANUAL TEST ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    user_id="570bfbe7-5474-4856-bf99-d5fac4b885a2"
    system_prompt = """
    
"""

    user_prompt="""
    Create a fresh new email for me, which follows to the leads of FMCG companies
"""

