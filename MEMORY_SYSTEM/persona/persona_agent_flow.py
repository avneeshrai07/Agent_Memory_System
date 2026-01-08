"""
Persona Agent Flow (End-to-End)
===============================

Purpose:
- Demonstrate the complete persona lifecycle in one place:
    1. Load persona from DB
    2. Inject persona into system prompt
    3. Call LLM to generate response
    4. Extract persona signals from interaction
    5. Merge + update persona in DB

This file is intentionally explicit and verbose.
No magic. No shortcuts.
"""
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import asyncio
import traceback

from dotenv import load_dotenv
load_dotenv()

from langchain_aws import ChatBedrock

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel
from MEMORY_SYSTEM.persona.persona_context_builder import build_persona_context
from MEMORY_SYSTEM.persona.persona_extractor import PersonaExtractor
from MEMORY_SYSTEM.persona.persona_merger import update_user_persona


# -------------------------------------------------------------------
# ENV + LLM INITIALIZATION
# -------------------------------------------------------------------

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_MODEL_REGION = "ap-southeast-2"
BEDROCK_MODEL_ID = "amazon.nova-lite-v1:0"

llm = ChatBedrock(
    model_id=BEDROCK_MODEL_ID,
    region_name=AWS_MODEL_REGION,
    temperature=0.2,
    max_tokens=2000,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


# -------------------------------------------------------------------
# RAW LLM CALL (GENERATION ONLY)
# -------------------------------------------------------------------

async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Simple generation call.
    Persona is already embedded in system_prompt.
    """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response.content


# -------------------------------------------------------------------
# MAIN ORCHESTRATION
# -------------------------------------------------------------------

async def build_user_persona_system_prompt(
    user_id: str,
    system_prompt: str
) -> str:
    """
    Build and return a persona-aware system prompt.
    """

    print("\n================ PERSONA CREATION FLOW START ================\n")
    print("[flow] user_id:", user_id)

    try:
        # ------------------------------------------------------
        # 1. LOAD PERSONA CONTEXT
        # ------------------------------------------------------

        persona_context = await build_persona_context(user_id)

        if persona_context:
            print("\n[flow] persona context loaded:")
            print(persona_context)
        else:
            print("\n[flow] no persona context found")

        personalised_system_prompt = f"""
        {system_prompt}

        The following describes the user's writing preferences and communication constraints:
        {persona_context}
                """.strip()

        print("\n================ PERSONA CREATION FLOW END ================\n")

        # ✅ RETURN HERE — NOTHING ELSE
        return personalised_system_prompt

    except Exception:
        print("\n[flow] ERROR OCCURRED")
        print(traceback.format_exc())
        raise



# ------------------------------------------------------
# 3. EXTRACT PERSONA FROM INTERACTION
# ------------------------------------------------------
async def learn_persona_from_interaction(
    user_id: str,
    user_prompt: str,
    llm
):
    try:
        print("\n================ PERSONA LEARNER FLOW START ================\n")
        persona_extractor = PersonaExtractor(llm)

        extracted_persona = await persona_extractor.extract(user_prompt)

        if extracted_persona is None:
            return

        await update_user_persona(
            user_id=user_id,
            incoming_persona=extracted_persona
        )

        print("[persona_learning] persona updated")
        print("\n================ PERSONA LEARNER FLOW START ================\n")
    except Exception:
        print("[persona_learning] ERROR")
        print(traceback.format_exc())



# -------------------------------------------------------------------
# MANUAL TEST ENTRY POINT
# -------------------------------------------------------------------
