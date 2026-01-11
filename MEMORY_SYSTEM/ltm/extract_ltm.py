# MEMORY_SYSTEM/ltm/extract_ltm.py

from typing import Dict, List
import traceback

from MEMORY_SYSTEM.llm.bedrock_structured import bedrock_structured_llm_call
from MEMORY_SYSTEM.ltm.ltm_fact_schema import LTMMemoryExtractionBatch
from MEMORY_SYSTEM.ltm.store_ltm import store_ltm_facts
from MEMORY_SYSTEM.ltm.store_episodic_ltm import store_episodic_ltm


# =====================================================
# LTM EXTRACTION (FACTUAL + EPISODIC)
# =====================================================
async def extract_ltm_facts(
    user_id: str,
    user_message: str,
    assistant_message: str
) -> Dict[str, List[Dict]]:
    """
    Extract both:
    - factual long-term memories (durable truths)
    - episodic long-term memories (referential / task continuity)

    This function:
    - extracts only (no decay, no prioritization here)
    - delegates storage decisions downstream
    """

    print("\n🧠 [LTM-EXTRACT] Starting extraction")

    # -------------------------------------------------
    # SYSTEM PROMPT (AUTHORITATIVE)
    # -------------------------------------------------
    system_prompt = """
You are a Long-Term Memory extraction engine for a conversational AI system.

Your task is to extract TWO DISTINCT TYPES of memory:

1. FACTUAL LTM:
   - Durable truths that remain useful across future sessions
   - Independent of the current conversation
   - Stable facts, constraints, preferences, expertise, validated outcomes

2. EPISODIC LTM:
   - Referential or narrative context needed to maintain continuity
   - Entity bindings ("the person we talked about")
   - Ongoing goals or active tasks
   - Active artifacts (emails, documents, plans)
   - Context that resolves phrases like:
     "that person", "the same email", "we discussed earlier"

--------------------------------------------------
STRICT GLOBAL RULES
--------------------------------------------------
- Do NOT summarize the conversation
- Do NOT restate assistant reasoning
- Do NOT invent information
- Do NOT store raw dialogue
- Do NOT include IDs, timestamps, or system fields
- Prefer omission over guessing

--------------------------------------------------
FACTUAL LTM RULES
--------------------------------------------------
ONLY extract if the information:
- Will be useful in future, unrelated sessions
- Can stand alone without conversational context
- Represents truth, constraint, preference, or expertise

DO NOT extract:
- Drafts, templates, or one-off artifacts
- Temporary tasks or wording
- Session-specific instructions

--------------------------------------------------
EPISODIC LTM RULES
--------------------------------------------------
Extract episodic context ONLY if it helps resolve continuity across turns.

Examples:
- Who "the famous personality" refers to
- What ongoing task the user is working on
- What artifact (email, plan) is currently active

Episodic memory is NOT a fact.
It is context.

--------------------------------------------------
CONFIDENCE RULES
--------------------------------------------------
- Explicit user statements → confidence.score = 1.0
- Clearly implied continuity → 0.8–0.9
- Weak inference → DO NOT EXTRACT

--------------------------------------------------
OUTPUT RULES
--------------------------------------------------
- Return ONLY valid JSON
- No markdown
- No commentary
- If nothing applies, return empty lists
"""

    # -------------------------------------------------
    # USER PROMPT
    # -------------------------------------------------
    user_prompt = f"""
Extract factual and episodic long-term memories from the following exchange.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_message}

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
Return a JSON object with exactly two keys:

1. "facts": list of factual LTM objects
2. "episodic": list of episodic LTM objects

FACTUAL MEMORY OBJECT:
- category: one of
  ["technical_context", "problem_domain", "constraint",
   "preference", "expertise", "validated_outcome", "learned_pattern"]
- topic: short label (3–6 words)
- fact: atomic factual statement
- importance: number from 1 to 10
- confidence:
    - score: 0.0–1.0
    - source: "explicit" | "implicit" | "derived" | "validated"

EPISODIC MEMORY OBJECT:
- context_type: one of
  ["entity_binding", "referential_alias",
   "ongoing_goal", "active_artifact",
   "conversation_focus", "role_assignment"]
- key: short identifier (e.g. "famous_personality", "current_goal")
- value: resolved meaning
- scope: "session" | "multi_turn" | "task"
- confidence:
    - score: 0.0–1.0
    - source: "explicit" | "implicit" | "inferred"

--------------------------------------------------
IMPORTANT
--------------------------------------------------
- It is valid to return empty lists for either field
- Do NOT wrap items in extra keys
"""

    # -------------------------------------------------
    # LLM CALL
    # -------------------------------------------------
    try:
        print("📨 [LTM-EXTRACT] Calling LLM")

        llm_response = await bedrock_structured_llm_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_structure=LTMMemoryExtractionBatch,
            model_dump=True,
        )

        if not llm_response:
            print("⚠️ [LTM-EXTRACT] Empty response from LLM")
            return {"facts": [], "episodic": []}

        print("[LTM-EXTRACT-RESPONSE]", llm_response)

        facts = llm_response.get("facts", [])
        episodic = llm_response.get("episodic", [])

        if not isinstance(facts, list) or not isinstance(episodic, list):
            print("❌ [LTM-EXTRACT] Invalid output shape")
            return {"facts": [], "episodic": []}

        print(f"✅ [LTM-EXTRACT] Extracted {len(facts)} factual memories")
        print(f"🧭 [LTM-EXTRACT] Extracted {len(episodic)} episodic contexts")

        # -------------------------------------------------
        # STORE FACTUAL LTM ONLY (for now)
        # -------------------------------------------------
        if facts:
            await store_ltm_facts(
                user_id=user_id,
                extracted_facts=facts,
                raw_context=user_message,
            )

        # NOTE:
        if episodic:
            await store_episodic_ltm(
                user_id,
                episodic,
                user_message
            )
        # Episodic storage, decay, and retrieval priority
        # will be wired in later stages as requested.

        return {
            "facts": facts,
            "episodic": episodic,
        }

    except Exception:
        print("❌ [LTM-EXTRACT] Extraction failed")
        traceback.print_exc()
        return {"facts": [], "episodic": []}
