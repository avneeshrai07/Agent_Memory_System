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
    - Persona signals NEVER enter learning cognition
    - Persona decisions ALWAYS include scope
    - Logging failures never block cognition
    - Decision shape is always valid
    """

    print(">>> ENTER run_cognition", flush=True)

    decisions: List[Dict[str, Any]] = []

    # --------------------------------------------------
    # Run cognition per signal (persona-aware)
    # --------------------------------------------------
    for signal in signal_candidates:
        safe_signal = dict(signal)

        try:
            # ==================================================
            # PERSONA SHORT-CIRCUIT (CRITICAL FIX)
            # ==================================================
            if safe_signal.get("epistemic_role") == "persona":
                decision = {
                    "action": "COMMIT",
                    "target": "persona",
                    "scope": [safe_signal["field"]],   # ✅ REQUIRED
                    "confidence": 1.0,
                    "reason": "explicit persona declaration",
                }

                decisions.append(decision)

                # Persona decisions are NOT logged as patterns
                continue

            # ==================================================
            # LEARNABLE SIGNAL → NORMAL COGNITION
            # ==================================================
            decision = await decide(safe_signal)

            # normalize decision shape (defensive)
            decision = {
                "action": decision.get("action", "REJECT"),
                "target": decision.get("target"),
                "scope": decision.get("scope", []),
                "confidence": float(decision.get("confidence", 0.0)),
                "reason": decision.get("reason"),
            }

            decisions.append(decision)

            # -----------------------------
            # Non-blocking pattern log
            # -----------------------------
            try:
                await log_pattern_decision(user_id, safe_signal, decision)
            except Exception as e:
                print(
                    "⚠️ pattern log failed:",
                    "signal =", safe_signal,
                    "decision =", decision,
                    "error =", repr(e),
                    flush=True,
                )

        except Exception as e:
            # Cognition itself failed — MUST be visible
            print(
                "❌ cognition decision failed:",
                "signal =", safe_signal,
                "error =", repr(e),
                flush=True,
            )

            decisions.append({
                "action": "REJECT",
                "target": None,
                "scope": [],
                "confidence": 0.0,
                "reason": "cognition_execution_failure",
            })

    print(">>> EXIT run_cognition", flush=True)
    return decisions
