# """
# Persona Extraction Prompts
# ==========================

# This file contains ALL prompt definitions used for persona extraction.
from MEMORY_SYSTEM.stm.stm_intent import STMIntent
from MEMORY_SYSTEM.llm.bedrock_structured import bedrock_structured_llm_call
import json
async def stm_intent_extractor_function(user_message):
    try:
        STM_INTENT_EXTRACTION_SYSTEM_PROMPT = """
            You are an intent interpreter for a state memory system.

    Your job is to analyze a user message and decide
    whether it represents an explicit, irreversible state change.

    State memory stores ONLY:
    - goals
    - decisions
    - constraints
    - approvals
    - rejections
    - direction changes
    - scope locks

    DO NOT extract:
    - questions
    - ideas
    - suggestions
    - brainstorming
    - exploration
    - temporary thoughts

    If the message does NOT clearly change state, set should_write = false.

    If it DOES, be precise and conservative.

            """

        user_prompt = f"""
        User message:
    <<<{user_message}>>>

    """

        intent = await bedrock_structured_llm_call(user_prompt=user_prompt, system_prompt=STM_INTENT_EXTRACTION_SYSTEM_PROMPT, output_structure=STMIntent, model_dump=True)
        
        print("[LLM_INTENT]     :", intent)
        return intent

    except Exception as e:
        print("[LLM_INTENT][ERROR]", str(e))
        raise


