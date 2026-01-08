# cognition/cognition_updater.py

from typing import List, Dict, Any

from MEMORY_SYSTEM.cognition.reasoning_policy import decide
from MEMORY_SYSTEM.cognition.cognition_model import CognitionModel
from MEMORY_SYSTEM.cognition.load_cognition import load_cognition_config
from MEMORY_SYSTEM.database.insert.log_pattern_decision import log_pattern_decision


async def run_cognition(
    signal_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    decisions: List[Dict[str, Any]] = []

    try:
        config = await load_cognition_config()
        cognition_model = CognitionModel(config)
    except Exception:
        cognition_model = CognitionModel({})

    for signal in signal_candidates:
        try:
            decision = await decide(signal, cognition_model)
            decisions.append(decision)

            # 🔒 SAFE FIRST CONSUMER: pattern log
            await log_pattern_decision(
                signal=signal,
                decision=decision,
            )

        except Exception:
            # Cognition must continue even if logging fails
            decisions.append({
                "action": "REJECT",
                "target": None,
                "scope": [],
                "confidence": 0.0,
                "reason": "cognition_execution_failure",
            })

    return decisions
