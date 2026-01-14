

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

from MEMORY_SYSTEM.runtime.background_worker import submit_background_task
from MEMORY_SYSTEM.context.build_cognition_context import build_epistemic_system_prompt
from MEMORY_SYSTEM.persona.persona_agent_flow import bring_user_persona
from MEMORY_SYSTEM.persona.persona_agent_flow import learn_persona_from_interaction
from MEMORY_SYSTEM.ltm.extract_ltm import extract_ltm_facts
from MEMORY_SYSTEM.ltm.retriever import retrieve_ltm_memories
from MEMORY_SYSTEM.ltm.context_builder import build_ltm_context
from MEMORY_SYSTEM.stm.stm_orchestrator import process_user_message
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
    user_prompt: str
):
    try:
        print("\n\n===================SYSTEM PROMPT START===================\n\n")
        print(system_prompt)
        print("\n\n===================SYSTEM PROMPT END===================\n\n")

        print("\n\n===================USER PROMPT START===================\n\n")
        print(user_prompt)
        print("\n\n===================USER PROMPT END===================\n\n")

        result = await process_user_message(
        user_id,
        user_prompt
    )
        print("result:  ", result)
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
        submit_background_task(
            learn_persona_from_interaction(user_id, user_prompt)
        )

        if response is None:
            return {"error": "bedrock returned null response"}

        agent_response = response.model_dump()
        # print("agent_response:  ",agent_response)
        print(agent_response.get('content'))
        agent_response_content = agent_response.get('content')

        submit_background_task(
            extract_ltm_facts(user_id, user_prompt, agent_response_content)
        )
        # print("agent_response_type", agent_response)
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
    You are an assistant operating with persistent state.

The following entries represent binding decisions, constraints, or directions.
They MUST guide your response.

IMPORTANT RULES:
- Use these entries to guide reasoning and choices.
- Do NOT mention or restate them unless the user explicitly asks "why".
- Do NOT explain past decisions unless asked.
- Do NOT say phrases like "as discussed earlier".
"""

    user_prompt="""
    can we try emails campaigns or something like that, or even a LinkedIn DM campaign?
"""
#     user_prompt = """
# write it in bullet points and keep it short
# """
    # user_prompt = """
    # i am a lead software developer at orbitaim, write the email again
    # """

    # user_prompt = """
    # i am a lead software developer at orbitaim, write the email again
    # """
#     user_prompt = """
#     keep emails short and in bullet points like before but add more action items
# """
#     user_prompt = """
#     I am Avneesh rai a software developer at orbitaim, write a blog for me
# """
    asyncio.run(
        bedrock_llm_call(
            user_id,
            system_prompt,
            user_prompt
        )
    )
