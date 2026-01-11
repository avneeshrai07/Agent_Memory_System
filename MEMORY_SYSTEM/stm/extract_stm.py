# MEMORY_SYSTEM/STM/extract_STM.py

from typing import Dict, Any, Optional
from MEMORY_SYSTEM.llm.bedrock_structured import bedrock_structured_llm_call
from MEMORY_SYSTEM.stm.stm_facts_schema import STMContext


async def extract_STM_facts(
    user_id: str,
    user_message: str,
    assistant_message: str
) -> Optional[Dict[str, Any]]:
    """
    Extract session-scoped STM updates from the latest user/assistant exchange.

    Returns:
        - Dict with STM fields to MERGE into existing STMContext
        - {} if no update
        - None if extraction failed
    """

    print("\n🧠 [STM-EXTRACT] Starting extraction")

    system_prompt = """
You are a Short-Term Memory (STM) extraction engine for a conversational AI system.

Your task is to extract ONLY session-scoped, temporary context needed to
continue the current conversation correctly.

STM is NOT long-term memory.

STRICT RULES:
- Extract ONLY what is relevant for the NEXT few turns.
- Do NOT extract durable facts meant for future sessions.
- Do NOT store historical preferences or long-term constraints.
- Do NOT summarize the conversation.
- Do NOT repeat the user or assistant messages verbatim.
- Do NOT infer intent beyond what is clearly stated or implied for this session.

STM EXISTS TO:
1. Track the user's immediate goal
2. Track the current problem-solving stage
3. Capture confirmed assumptions and constraints (session-scoped)
4. Record decisions already made to avoid repetition
5. Identify open questions that still need answers
6. Preserve temporary preferences that apply only to this session

FAILURE MODE:
- If nothing new or relevant exists, return an empty JSON object {}.

OUTPUT FORMAT:
- Return ONLY valid JSON
- No markdown
- No explanations
- No commentary
"""

    user_prompt = f"""
Extract Short-Term Memory (STM) updates from the following exchange.

USER MESSAGE:
\"\"\"
{user_message}
\"\"\"

ASSISTANT RESPONSE:
\"\"\"
{assistant_message}
\"\"\"

Return a JSON object with ANY of the following OPTIONAL fields
(omit fields that do not apply):

{{
  "current_goal": string | null,
  "stage": "exploration" | "diagnosis" | "solution" | "validation" | null,
  "confirmed_constraints": [string],
  "assumptions": [string],
  "decisions_made": [string],
  "open_questions": [string],
  "temporary_preferences": [string]
}}

If nothing should be updated, return {{}}.
"""

    try:
        print("📨 [STM-EXTRACT] Calling LLM")

        llm_response = await bedrock_structured_llm_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_structure=STMContext,
            model_dump=True
        )

        if not llm_response:
            print("⚠️ [STM-EXTRACT] Empty LLM response")
            return {}

        if not isinstance(llm_response, dict):
            print("❌ [STM-EXTRACT] LLM output is not a dict")
            return None

        # STM extractor returns a PARTIAL STMContext update
        print("✅ [STM-EXTRACT] STM update extracted:", llm_response)
        return llm_response

    except Exception as e:
        print(f"❌ [STM-EXTRACT] Extraction failed: {e}")
        return None
