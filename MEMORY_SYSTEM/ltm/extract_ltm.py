# MEMORY_SYSTEM/ltm/extract_ltm.py

import json
from typing import List, Dict
from MEMORY_SYSTEM.llm.extraction import call_extraction_llm


async def extract_ltm_facts(
    user_message: str,
    assistant_message: str
) -> List[Dict]:
    print("\n🧠 [LTM-EXTRACT] Starting extraction")

    extraction_prompt = f"""
Extract factual statements that should be remembered for future interactions.

Rules:
- Extract only explicit facts, constraints, goals, corrections, or validations.
- One fact per item.
- No opinions or inferred traits.
- If nothing is worth remembering, return an empty array.

Conversation:
User: {user_message}
Assistant: {assistant_message}

Output JSON only:
[
  {{
    "fact": "...",
    "memory_type": "...",
    "confidence_score": 0.0,
    "semantic_topic": "..."
  }}
]
"""

    try:
        print("📨 [LTM-EXTRACT] Calling LLM")
        llm_response = await call_extraction_llm(extraction_prompt)

        try:
            extracted = json.loads(llm_response)
        except Exception:
            print("❌ [LTM-EXTRACT] Invalid JSON from LLM")
            print(llm_response)
            return []

        if not isinstance(extracted, list):
            print("❌ [LTM-EXTRACT] LLM output is not a list")
            return []

        print(f"✅ [LTM-EXTRACT] Extracted {len(extracted)} facts")
        return extracted

    except Exception as e:
        print(f"❌ [LTM-EXTRACT] Extraction failed: {e}")
        return []
