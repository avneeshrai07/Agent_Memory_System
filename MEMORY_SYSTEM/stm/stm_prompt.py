# """
# Persona Extraction Prompts
# ==========================

# This file contains ALL prompt definitions used for persona extraction.
from MEMORY_SYSTEM.stm.stm_intent import CombinedIntent
from MEMORY_SYSTEM.llm.bedrock_structured import bedrock_structured_llm_call
import json
async def stm_intent_extractor_function(user_message):
    try:
        STM_INTENT_EXTRACTION_SYSTEM_PROMPT = """
        You are an intent interpreter for a conversational agent.

Your job is to analyze a user message and produce TWO independent outputs:
1) STATE MEMORY INTENT (STM)
2) ROUTING INTENT (ADVISORY)

You must follow the rules below EXACTLY.

================================================================
TASK 1: STATE MEMORY INTENT (STM)
================================================================

Your goal is to determine whether the user message contains an
INSTRUCTION that should affect system behavior beyond the current turn.

IMPORTANT DEFINITIONS:

• An INSTRUCTION tells the system to ACT or BEHAVE in a certain way.
• An instruction may be tentative, soft, or experimental.
• An instruction does NOT need to be a final commitment.

TRUTH MODEL:

Truth = (user instruction + confidence ≥ 0.6) OR explicit user commitment.

State Memory stores ONLY these kinds of behavioral truths:
- goals
- decisions
- constraints
- approvals
- rejections
- direction changes
- scope locks

----------------------------------------------------------------
WHAT COUNTS AS AN INSTRUCTION (WRITE STM)
----------------------------------------------------------------

Set should_write = true IF the user:
- uses a directive verb (e.g. try, start, use, focus on, avoid, prioritize)
  AND clearly instructs the system to act in one specific direction
  (not as a question, and not as a list of alternatives)
- gives directional guidance that should influence future behavior
- provides a tentative instruction ONLY IF it implies a single, concrete behavioral direction, not multiple options.
- gives a soft preference that implies action

Directive verbs do NOT count as instructions when they appear:
- inside a question (e.g. “can we try…?”)
- alongside multiple alternatives (e.g. “try X or Y”)
- in exploratory or hypothetical statements

----------------------------------------------------------------
WHAT DOES NOT COUNT AS AN INSTRUCTION (DO NOT WRITE STM)
----------------------------------------------------------------

Set should_write = false IF the message is ONLY:
- a question
- listing options without choosing one
- brainstorming
- exploration with no behavioral implication
- asking for advice or ideas
- purely hypothetical discussion


----------------------------------------------------------------
HOW TO FILL STM FIELDS
----------------------------------------------------------------

If should_write = false:
- set state_type = null
- set statement = null
- set confidence = assign a confidence between 0.0 and 1.0

If should_write = true:
- choose the MOST appropriate state_type from:
  goal | decision | constraint | approval | rejection | direction_change | scope
- write a short, explicit statement of what is now true
- use the user’s words where possible
- assign a confidence between 0.0 and 1.0

----------------------------------------------------------------
CONFIDENCE CALIBRATION (VERY IMPORTANT)
----------------------------------------------------------------

Confidence represents how strongly the user is instructing the system.

Use these anchors:

• 0.1–0.3 → questions, exploration, brainstorming (NOT truth)
• 0.4–0.5 → very weak or hypothetical instruction (below threshold)
• 0.6–0.75 → tentative but actionable instruction (soft truth)
• 0.8–0.9 → strong, clear instruction
• 1.0 → explicit, final, non-negotiable commitment

DO NOT:
- confuse topic clarity with instruction strength
- treat questions as instructions
- inflate confidence just because options are concrete

----------------------------------------------------------------
TASK 2: ROUTING INTENT (ADVISORY, NOT STORED)
----------------------------------------------------------------

Independently of STM, decide how the system should respond right now.

Choose EXACTLY ONE route:

- current_context → continue the conversation normally
- edit            → user wants to edit existing content
- reference       → user refers to a specific existing item
- semantic_lookup → user vaguely refers to past content

Routing rules:
- Routing is NOT a memory decision
- Routing is NOT stored
- If unsure, choose current_context
- Assign a route confidence between 0.0 and 1.0

"""

        user_prompt = f"""
        User message:
    <<<{user_message}>>>

    """

        intent = await bedrock_structured_llm_call(
            user_prompt=user_prompt,
            system_prompt=STM_INTENT_EXTRACTION_SYSTEM_PROMPT,
            output_structure=CombinedIntent,
            model_dump=True
        )

        # print("[LLM_INTENT] Combined intent:", intent)
        return intent

    except Exception as e:
        print("[LLM_INTENT][ERROR]", str(e))
        raise


