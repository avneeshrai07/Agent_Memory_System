# MEMORY_SYSTEM/ltm/extract_ltm.py

import json
from typing import List, Dict
from MEMORY_SYSTEM.llm.bedrock_structured import bedrock_structured_llm_call
from MEMORY_SYSTEM.ltm.ltm_fact_schema import LTMMemoryExtractionBatch
from MEMORY_SYSTEM.ltm.store_ltm import store_ltm_facts
async def extract_ltm_facts(
    user_id: str,
    user_message: str,
    assistant_message: str
) -> List[Dict]:
    print("\n🧠 [LTM-EXTRACT] Starting extraction")

    system_prompt = """
You are a memory extraction engine for a conversational AI system.

Your task is to extract ONLY durable, reusable information that should be stored
in Long-Term Memory (LTM) to improve future interactions.

STRICT RULES:
- Extract FACTS, not conversation.
- Do NOT summarize the discussion.
- Do NOT restate the assistant’s reasoning.
- Do NOT infer intent unless explicitly stated.
- Do NOT invent information.
- Do NOT store temporary confusion, questions, or session-only context.

ONLY extract information that:
1. Will still be useful in a future session
2. Can stand alone without conversational context
3. Affects future reasoning, constraints, preferences, or advice

WHAT TO EXTRACT:
- Explicit technical facts (systems, scale, metrics)
- Hard constraints (cannot do X, must do Y)
- Stable preferences (format, depth, style) if explicit
- Validated outcomes (what worked / did not work)
- Repeated or clearly stated problem domains
- Clear expertise indicators

WHAT NOT TO EXTRACT:
- Single-use questions
- Ongoing troubleshooting steps
- Hypotheses or guesses
- Assistant suggestions unless validated by the user
- Anything that only matters inside this session

CONFIDENCE RULES:
- Explicit user statements → confidence = 1.0
- Validated outcomes → confidence = 0.9
- Repeated signals → confidence = 0.75–0.85
- Ambiguous signals → DO NOT EXTRACT

OUTPUT FORMAT:
- Return ONLY valid JSON
- No markdown
- No explanation text
- No commentary

If no durable memories exist, return an empty JSON array [].

    """

    user_prompt = f"""
Extract long-term memories from the following exchange.

USER MESSAGE:

{user_message}


ASSISTANT RESPONSE:

{assistant_message}


INSTRUCTIONS:
- Extract 0 to 5 memories
- Each memory must be atomic and independently reusable
- Phrase memories as factual statements, not summaries
- Prefer precision over completeness

For each extracted memory, return an object with:

- category: one of
  ["technical_context", "problem_domain", "constraint", "preference", "expertise", "validated_outcome", "learned_pattern"]

- topic: short label (3–6 words)

- fact: a precise factual statement

- importance: number from 1 to 10
  (10 = critical, long-lasting, repeatedly relevant)

- confidence:
  (
    "score": number from 0.0 to 1.0,
    "source": one of ["explicit", "validated", "implicit"]
  )

Return the result as a JSON array.
"""
    
    try:
        print("📨 [LTM-EXTRACT] Calling LLM")
        llm_response = await bedrock_structured_llm_call(
            system_prompt=system_prompt, 
            user_prompt=user_prompt, 
            output_structure=LTMMemoryExtractionBatch,
            model_dump=True)
        
        
        print("[LTM-EXTRACT-RESPONSE]   ", llm_response)
        extracted = llm_response.get("facts",[])
        print("[LTM-EXTRACTED-FACTS]   ", extracted)
        
        if not isinstance(extracted, list):
            print("❌ [LTM-EXTRACT] LLM output is not a list")
            return []

        print(f"✅ [LTM-EXTRACT] Extracted {len(extracted)} facts")
        
        await store_ltm_facts(user_id=user_id,extracted_facts= extracted, raw_context=user_message)

    except Exception as e:
        print(f"❌ [LTM-EXTRACT] Extraction failed: {e}")
        return []
