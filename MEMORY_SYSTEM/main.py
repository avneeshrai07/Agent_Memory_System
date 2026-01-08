

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
from MEMORY_SYSTEM.persona.persona_agent_flow import learn_persona_from_interaction
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
        # --------------------------------------------------
        # BACKGROUND PERSONA LEARNING (NON-BLOCKING)
        # --------------------------------------------------
        task = asyncio.create_task(
            learn_persona_from_interaction(
                user_id,
                user_prompt
            ),
            name=f"process_memory:{user_id}",
        )

        register_task(task)


        if response is None:
            return {"error": "bedrock returned null response"}

        agent_response = response.model_dump()
        print("agent_response:  ",agent_response)
        await asyncio.sleep(30)
        # return agent_response['content']
        print("agent_response_type", agent_response)
        return agent_response

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }



# -------------------------------------------------------------------
# MANUAL TEST ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    user_id="test_user_004"
    system_prmopt = """
    You are a professional AI writing assistant.
"""

#     user_prompt="""
# Can you write me an email on prefabs design write in a professional, analytical tone. Keep responses short and structured in paragraphs. Avoid casual language completely.
# """
    user_prompt = """
cMy name is Harsh Joshi.

I am the Chief Marketing Officer at Orbitaim.
I work primarily in the marketing function and I am a decision maker for content and go-to-market strategy.

Orbitaim is a technology company.
The registered name of the company is Orbit AI Marketing Solutions LLP.
The company operates in the Technology industry.
We are a startup with around 11–50 employees.
Our headquarters is in Sector-142, Noida.
Our website is https://orbit.orbitaim.io.

Orbitaim follows a B2B SaaS business model.
Our primary customers are startups and small to medium-sized businesses.
Our flagship product is called Orbit, which is a software product.

For content creation:
- The primary goal is to persuade.
- The desired reader action is to book a demo.
- Success is measured by demo bookings.

I usually want content written as emails.
Use paragraphs, not bullet points.
Keep the length short.

The target audience is executives in the SaaS domain.
They are experts.

Maintain a professional tone.
Use third person voice.
The emotional intensity should be confident.

Adopt an analytical writing style.
Prefer short sentences.

Write in English.
Use a professional level of complexity.
Jargon is allowed.

Do not exaggerate claims.
Avoid informal language.
"""
    asyncio.run(
        bedrock_llm_call(
            user_id,
            system_prmopt,
            user_prompt
        )
    )
