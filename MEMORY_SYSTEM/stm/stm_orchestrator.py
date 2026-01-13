from MEMORY_SYSTEM.stm.stm_prompt import stm_intent_extractor_function


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

        stm_entry = await stm_intent_extractor_function(
            message
        )

        if stm_entry:
            print("[ORCHESTRATOR] STM updated:", stm_entry.stm_id)
        else:
            print("[ORCHESTRATOR] No STM update")

        return {
            "stm_written": stm_entry is not None,
            "stm_entry": stm_entry,
            "raw_message": message
        }

    except Exception as e:
        print("[ORCHESTRATOR][ERROR]", str(e))
        raise
