# cognition/cognition_updater.py

from typing import List, Dict, Any

from MEMORY_SYSTEM.cognition.reasoning_policy import decide
from MEMORY_SYSTEM.cognition.cognition_model import CognitionModel
from MEMORY_SYSTEM.cognition.load_cognition import load_cognition_config
from MEMORY_SYSTEM.database.insert.log_pattern_decision import log_pattern_decision


async def run_cognition(
    user_id: str,
    signal_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run cognition over signal candidates and return decisions.

    Guarantees:
    - Cognition never mutates incoming signals
    - Every signal produces exactly one decision
    - Logging failures never block cognition
    - Decision shape is always valid
    """

    decisions: List[Dict[str, Any]] = []

    # --------------------------------------------------
    # Load cognition config (fail-safe)
    # --------------------------------------------------
    try:
        config = await load_cognition_config()
        cognition_model = CognitionModel(config)
    except Exception:
        # Absolute fallback: empty model
        cognition_model = CognitionModel({})

    # --------------------------------------------------
    # Run cognition per signal (isolated execution)
    # --------------------------------------------------
    for signal in signal_candidates:
        # Defensive copy: cognition must not mutate source signals
        safe_signal = dict(signal)

        try:
            decision = await decide(safe_signal, cognition_model)

            # Enforce decision shape invariants
            decision = {
                "action": decision.get("action", "REJECT"),
                "target": decision.get("target"),
                "scope": decision.get("scope", []),
                "confidence": float(decision.get("confidence", 0.0)),
                "reason": decision.get("reason"),
            }

            decisions.append(decision)

            # --------------------------------------------------
            # SAFE FIRST CONSUMER: pattern log (non-blocking)
            # --------------------------------------------------
            try:
                await log_pattern_decision(user_id, safe_signal, decision)
            except Exception:
                # Logging failure must NEVER affect cognition
                pass

        except Exception:
            # Absolute cognition failure → explicit reject
            decisions.append({
                "action": "REJECT",
                "target": None,
                "scope": [],
                "confidence": 0.0,
                "reason": "cognition_execution_failure",
            })

    return decisions
