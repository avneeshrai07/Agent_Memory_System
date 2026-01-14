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

        stm_intent = intent["stm"]
        route_intent = intent["route"]

        print("[ORCHESTRATOR] STM intent:", stm_intent)
        print("[ORCHESTRATOR] Route intent:", route_intent)

        # ----------------------------------
        # 2. Routing decision (ALWAYS)
        # ----------------------------------
        route = route_intent["route"]
        route_confidence = route_intent["confidence"]

        # ----------------------------------
        # 3. STM gatekeeper (CONDITIONAL)
        # ----------------------------------
        stm_entry = None

        if approve_stm_intent(stm_intent):
            print("[ORCHESTRATOR] STM write approved")

            stm_entry = await commit_stm_intent(
                user_id=user_id,
                intent=stm_intent
            )

            print("[ORCHESTRATOR] STM updated:", stm_entry["stm_id"])
        else:
            print("[ORCHESTRATOR] STM write rejected by gatekeeper")

        # ----------------------------------
        # 4. Return full orchestration result
        # ----------------------------------
        return {
            "route": route,
            "route_confidence": route_confidence,
            "stm_written": stm_entry is not None,
            "stm_entry": stm_entry,
            "raw_message": message
        }

    except Exception as e:
        print("[ORCHESTRATOR][ERROR]", str(e))
        raise
