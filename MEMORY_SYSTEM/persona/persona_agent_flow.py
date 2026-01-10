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
import traceback
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_aws import ChatBedrock

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel
from MEMORY_SYSTEM.persona.persona_context_builder import build_persona_context
# from MEMORY_SYSTEM.persona.persona_extractor import persona_extractor_llm_call
from MEMORY_SYSTEM.persona.persona_prompts import persona_extractor_function
from MEMORY_SYSTEM.persona.persona_merger import update_user_persona
from MEMORY_SYSTEM.persona.persona_adapters import (
    persona_to_signals,
    project_persona_by_decisions,
)
from MEMORY_SYSTEM.cognition.cognition_updater import run_cognition
from MEMORY_SYSTEM.cognition.signal_frequency import enrich_signal_frequency

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

def print_signals_with_decisions(signals, decisions, flush=True):
    print("\n=== SIGNALS + COGNITION DECISIONS ===", flush=flush)

    for idx, (signal, decision) in enumerate(zip(signals, decisions), start=1):
        print(f"\n[{idx}] {signal.get('category', '').upper()} :: {signal.get('field')}", flush=flush)

        # --- Signal ---
        print("  SIGNAL", flush=flush)
        print(f"    value        : {signal.get('value')}", flush=flush)
        print(f"    base_conf    : {signal.get('base_confidence')}", flush=flush)
        print(f"    source       : {signal.get('source')}", flush=flush)
        print(f"    frequency    : {signal.get('frequency')}", flush=flush)

        # --- Decision ---
        print("  DECISION", flush=flush)
        print(f"    action       : {decision.get('action')}", flush=flush)
        print(f"    target       : {decision.get('target')}", flush=flush)
        print(f"    confidence   : {decision.get('confidence')}", flush=flush)
        print(f"    reason       : {decision.get('reason')}", flush=flush)

        # --- Debug warnings (very important for you) ---
        if signal.get("source") == "extracted" and signal.get("base_confidence") == 1.0:
            print("  ⚠ WARNING", flush=flush)
            print("    extracted signal has confidence=1.0 (persona bootstrap bug)", flush=flush)

    print("\n=== END SIGNALS + DECISIONS ===\n", flush=flush)


def print_persona_human_readable(persona, flush=True):
    print("\n================ USER PERSONA (HUMAN READABLE) ================\n", flush=flush)

    def print_block(title: str, block):
        if block is None:
            return

        print(f"▶ {title}", flush=flush)
        for field, value in block.model_dump().items():
            if value in (None, "N/A", [], {}):
                continue
            print(f"  - {field}: {value}", flush=flush)
        print("", flush=flush)

    # --------------------------------------------------
    # CORE IDENTITY
    # --------------------------------------------------
    print_block("User Identity", persona.user_identity)

    # --------------------------------------------------
    # COMPANY CONTEXT
    # --------------------------------------------------
    print_block("Company Profile", persona.company_profile)
    print_block("Company Business", persona.company_business)
    print_block("Company Products", persona.company_products)
    print_block("Company Brand", persona.company_brand)

    # --------------------------------------------------
    # OBJECTIVE & CONTENT
    # --------------------------------------------------
    print_block("Objective", persona.objective)
    print_block("Content Format", persona.content_format)
    print_block("Audience", persona.audience)

    # --------------------------------------------------
    # STYLE & LANGUAGE
    # --------------------------------------------------
    print_block("Tone", persona.tone)
    print_block("Writing Style", persona.writing_style)
    print_block("Language", persona.language)

    # --------------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------------
    print_block("Constraints", persona.constraints)

    print("================ END USER PERSONA =================\n", flush=flush)

# ------------------------------------------------------
# 3. EXTRACT PERSONA FROM INTERACTION
# ------------------------------------------------------
# persona/persona_agent_flow.py

async def learn_persona_from_interaction(user_id: str, user_prompt: str):
    print(">>> ENTER learn_persona_from_interaction", flush=True)

    try:
        extracted_persona = await persona_extractor_function(user_prompt)
        print("extracted_persona:", extracted_persona, flush=True)

        signals = persona_to_signals(extracted_persona)
        # print("signals_Evidence:", signals, flush=True)

        signals = await enrich_signal_frequency(user_id, signals)
        

        print(">>> ABOUT TO RUN COGNITION", flush=True)
        decisions = await run_cognition(user_id, signals)
        print_signals_with_decisions(signals, decisions)

        filtered_persona = project_persona_by_decisions(extracted_persona, decisions)
        print("\n================ FILTERED PERSONA ================\n")
        print_persona_human_readable(filtered_persona)

        if filtered_persona:
            print(">>> ABOUT TO UPDATE DB", flush=True)
            await update_user_persona(user_id, filtered_persona)
            print(">>> DB UPDATED", flush=True)
        else:
            print(">>> NOTHING TO PERSIST", flush=True)

    except Exception as e:
        print("❌ persona learner crashed:", e, flush=True)

    print("<<< EXIT learn_persona_from_interaction", flush=True)
