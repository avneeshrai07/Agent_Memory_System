from MEMORY_SYSTEM.stm.stm_prompt import stm_intent_extractor_function
from MEMORY_SYSTEM.stm.stm_intent_gatekeeper import approve_stm_intent
from MEMORY_SYSTEM.stm.stm_repository import commit_stm_intent
async def process_user_message(
    user_id: str,
    message: str
):
    """
    Single entry point for processing a user message.
    """

    try:
        print("\n[ORCHESTRATOR] New user message")
        print("[ORCHESTRATOR] Message:", message)

        # ----------------------------------
        # 1. LLM intent interpretation
        # ----------------------------------
        intent = await stm_intent_extractor_function(message)

        print("[ORCHESTRATOR] LLM intent:", intent)

        # ----------------------------------
        # 2. Gatekeeper (authority decision)
        # ----------------------------------
        if not approve_stm_intent(intent):
            print("[ORCHESTRATOR] STM write rejected by gatekeeper")
            return {
                "stm_written": False,
                "stm_entry": None,
                "raw_message": message
            }

        # ----------------------------------
        # 3. STM commit (authoritative)
        # ----------------------------------
        stm_entry = await commit_stm_intent(
            user_id=user_id,
            intent=intent,
        )

        print("[ORCHESTRATOR] STM updated:", stm_entry["stm_id"])

        return {
            "stm_written": True,
            "stm_entry": stm_entry,
            "raw_message": message
        }

    except Exception as e:
        print("[ORCHESTRATOR][ERROR]", str(e))
        raise
