# MEMORY_SYSTEM/ltm/extract_ltm.py

import json
from typing import List, Dict
from MEMORY_SYSTEM.llm.bedrock_structured import bedrock_structured_llm_call
from MEMORY_SYSTEM.ltm.ltm_fact_schema import LTMFactList
from MEMORY_SYSTEM.ltm.store_ltm import store_ltm_facts
async def extract_ltm_facts(
    user_id: str,
    user_message: str,
    assistant_message: str
) -> List[Dict]:
    print("\n🧠 [LTM-EXTRACT] Starting extraction")

    system_prompt = """
You are a candidate extraction engine for a long-term memory system.

Your job is to IDENTIFY POSSIBLE long-term memory facts
explicitly mentioned or clearly stated by the USER.

IMPORTANT:
- You are NOT responsible for deciding truth or permanence.
- You are allowed to normalize wording to create clean, atomic facts.
- You are allowed to restate user statements more clearly.
- Your output will be VERIFIED by another system later.

ALLOWED:
- Normalize tense, grammar, and phrasing.
- Split compound statements into atomic facts.
- Convert instructions or declarations into factual form.
- Extract preferences, technical context, constraints, goals, validations, and corrections.

DISALLOWED:
- Do NOT invent new information.
- Do NOT use background knowledge.
- Do NOT add facts that are not grounded in the user message.
- Do NOT infer unstated technologies, budgets, architectures, or preferences.

IMPORTANT BEHAVIOR:
- If the user message contains durable decisions, configurations, or preferences,
  extract them even if they require light rephrasing.
- If the message contains only temporary status updates or plans,
  it is acceptable to return no facts.

    """

    user_prompt = f"""
Extract POSSIBLE long-term memory facts from the USER message below.

USER MESSAGE:

{user_message}


ASSISTANT MESSAGE (context only, do not extract from):

{assistant_message}


INSTRUCTIONS:
- Identify statements that could represent durable preferences, constraints,
  technical context, goals, validations, or corrections.
- You MAY normalize wording to make facts clearer and atomic.
- You MUST NOT invent information not present in the user message.
- You MUST NOT rely on assumptions or typical patterns.

For each candidate fact, return:
- fact: a clear, normalized statement
- memory_type: one of ["technical_context", "user_preference", "constraint", "goal", "validation", "correction"]
- semantic_topic: short label (2–4 words) or null
- confidence_score: 0.0–1.0 based on how explicitly the user stated it

"""
    try:
        print("📨 [LTM-EXTRACT] Calling LLM")
        llm_response = await bedrock_structured_llm_call(
            system_prompt=system_prompt, 
            user_prompt=user_prompt, 
            output_structure=LTMFactList,
            model_dump=True)
        
        print("[LTM-EXTRACT-RESPONSE]   ", llm_response)
        print("[LTM-EXTRACT-RESPONSE]_type   ", type(llm_response))
        extracted = llm_response.get("facts",[])
        # try:
        #     extracted = json.loads(llm_response)
        # except Exception:
        #     print("❌ [LTM-EXTRACT] Invalid JSON from LLM")
        #     print(llm_response)
        #     return []

        if not isinstance(extracted, list):
            print("❌ [LTM-EXTRACT] LLM output is not a list")
            return []

        print(f"✅ [LTM-EXTRACT] Extracted {len(extracted)} facts")
        
        await store_ltm_facts(user_id=user_id,extracted_facts= extracted, raw_context=user_message)

    except Exception as e:
        print(f"❌ [LTM-EXTRACT] Extraction failed: {e}")
        return []
