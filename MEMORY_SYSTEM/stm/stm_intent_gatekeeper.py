def approve_stm_intent(intent: dict) -> bool:
    try:
        print("[STM_GATE] Validating STM intent")

        # 1. Must be an instruction
        if not intent.get("should_write"):
            print("[STM_GATE] Rejected: should_write = false")
            return False

        # 2. Must specify what changed
        if not intent.get("state_type") or not intent.get("statement"):
            print("[STM_GATE] Rejected: missing state_type or statement")
            return False

        # 3. Confidence threshold (soft truth allowed)
        confidence = intent.get("confidence")
        if confidence is None:
            print("[STM_GATE] Rejected: confidence missing")
            return False

        if confidence < 0.6:
            print(f"[STM_GATE] Rejected: confidence too low ({confidence})")
            return False

        print(
            f"[STM_GATE] Approved STM intent "
            f"(confidence={confidence})"
        )
        return True

    except Exception as e:
        print("[STM_GATE][ERROR]", str(e))
        raise
