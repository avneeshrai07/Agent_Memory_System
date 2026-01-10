

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
from MEMORY_SYSTEM.persona.persona_agent_flow import build_user_persona_system_prompt
from MEMORY_SYSTEM.persona.persona_agent_flow import learn_persona_from_interaction
from MEMORY_SYSTEM.ltm.extract_ltm import extract_ltm_facts
from MEMORY_SYSTEM.ltm.retriever import retrieve_ltm_memories
from MEMORY_SYSTEM.ltm.context_builder import build_ltm_context
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
    base_system_prompt: str,
    user_prompt: str
):
    try:
        try:
            epistemic_system_prompt = build_epistemic_system_prompt(base_system_prompt)
            final_system_prompt = await build_user_persona_system_prompt(user_id, epistemic_system_prompt)
        except Exception:
            final_system_prompt = base_system_prompt



        try:
            memories = await retrieve_ltm_memories(user_id, user_prompt)
            ltm_context = build_ltm_context(memories)

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
        submit_background_task(
            await learn_persona_from_interaction(user_id, user_prompt)
        )

        if response is None:
            return {"error": "bedrock returned null response"}

        agent_response = response.model_dump()
        # print("agent_response:  ",agent_response)
        print(agent_response.get('content'))
        agent_response_content = agent_response.get('content')

        submit_background_task(
            await extract_ltm_facts(user_id, user_prompt, agent_response_content)
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
    user_id="test_user_011"
    system_prmopt = """
    You are a professional AI writing assistant.
"""

    user_prompt="""
    I’m not completely sure how to describe myself yet, but I’ll try.

I think I’m mostly an individual developer, though sometimes I also feel more like a system architect. I’ve worked with software for a while now — maybe around 8–10 years — but I don’t always work in the same area.

Lately, I’ve been experimenting with AI-related things, especially around agents and memory, but I wouldn’t say I’m an expert. I’m still figuring things out and learning as I go.

I might be working on something like a startup, or at least an early-stage project. It’s not a formal company yet, and a lot of details like company size or industry are still unclear.

My main goal right now is mostly to explore and understand how AI agent systems work, and maybe try implementing some ideas if they make sense. I don’t have very strict success criteria yet — it’s more about learning than achieving a specific outcome.

If I create content, it’s usually for developers like me, but again, that’s not fixed. The tone I prefer is generally technical, but I’m flexible and still experimenting with what works best.

Overall, many of these things are tentative, and I expect they might change as I learn more.
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
            system_prmopt,
            user_prompt
        )
    )
