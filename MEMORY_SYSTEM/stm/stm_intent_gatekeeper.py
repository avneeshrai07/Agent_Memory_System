from MEMORY_SYSTEM.stm.stm_intent import STMIntent


def approve_stm_intent(intent: STMIntent) -> bool:
    try:
        print("[STM_GATE] Validating STM intent")

        if not intent.should_write:
            print("[STM_GATE] Rejected: should_write = false")
            return False

        if not intent.state_type or not intent.statement:
            print("[STM_GATE] Rejected: missing state_type or statement")
            return False

        if intent.confidence is None or intent.confidence < 0.85:
            print("[STM_GATE] Rejected: confidence too low")
            return False

        print("[STM_GATE] Intent approved")
        return True

    except Exception as e:
        print("[STM_GATE][ERROR]", str(e))
        raise
