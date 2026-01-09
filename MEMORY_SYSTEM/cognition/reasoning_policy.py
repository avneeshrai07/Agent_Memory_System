# cognition/reasoning_policy.py

from typing import Dict, Any
from MEMORY_SYSTEM.cognition.cognition_model import CognitionModel


async def decide(
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Block-aligned cognition decision engine.

    Rules:
    - Implicit learning everywhere
    - Promotion after frequency >= 2 (>=3 for constraints)
    - Confidence only prevents hallucination
    """

    try:
        field = signal.get("field")
        category = signal.get("category")
        frequency = int(signal.get("frequency", 1))
        base_confidence = float(signal.get("base_confidence", 0.0))

        # -------------------------------------------------
        # GLOBAL SAFETY GATE
        # -------------------------------------------------
        if base_confidence < 0.80:
            return {
                "action": "REJECT",
                "target": None,
                "scope": [],
                "confidence": 0.0,
                "reason": "low confidence signal rejected",
            }

        # =================================================
        # 1. USER IDENTITY
        # =================================================
        if category == "identity":
            if frequency >= 2:
                return {
                    "action": "PARTIAL_COMMIT",
                    "target": "persona",
                    "scope": [field],
                    "confidence": 0.85,
                    "reason": "implicit repetition (identity)",
                }

            return {
                "action": "PROVISIONAL_COMMIT",
                "target": "runtime_only",
                "scope": [field],
                "confidence": 0.60,
                "reason": "provisional identity signal",
            }

        # =================================================
        # 2. ORGANIZATION / COMPANY
        # =================================================
        if category == "organization":
            if frequency >= 2:
                return {
                    "action": "PARTIAL_COMMIT",
                    "target": "persona",
                    "scope": [field],
                    "confidence": 0.85,
                    "reason": "implicit repetition (organization)",
                }

            return {
                "action": "PROVISIONAL_COMMIT",
                "target": "runtime_only",
                "scope": [field],
                "confidence": 0.60,
                "reason": "provisional organization context",
            }

        # =================================================
        # 3. PREFERENCES / STYLE / FORMAT / AUDIENCE
        # =================================================
        if category == "preference":
            if frequency >= 1:
                return {
                    "action": "PARTIAL_COMMIT",
                    "target": "persona",
                    "scope": [field],
                    "confidence": 0.85,
                    "reason": "implicit repetition (preference/style)",
                }

            return {
                "action": "PROVISIONAL_COMMIT",
                "target": "runtime_only",
                "scope": [field],
                "confidence": 0.60,
                "reason": "provisional preference",
            }

        # =================================================
        # 4. LANGUAGE
        # =================================================
        if category == "language":
            if frequency >= 1:
                return {
                    "action": "PARTIAL_COMMIT",
                    "target": "persona",
                    "scope": [field],
                    "confidence": 0.85,
                    "reason": "implicit repetition (language)",
                }

            return {
                "action": "PROVISIONAL_COMMIT",
                "target": "runtime_only",
                "scope": [field],
                "confidence": 0.60,
                "reason": "provisional language preference",
            }

        # =================================================
        # 5. CONSTRAINTS
        # =================================================
        if category == "constraint":
            if frequency >= 3:
                return {
                    "action": "PARTIAL_COMMIT",
                    "target": "persona",
                    "scope": [field],
                    "confidence": 0.85,
                    "reason": "repeated constraint accepted",
                }

            return {
                "action": "PROVISIONAL_COMMIT",
                "target": "runtime_only",
                "scope": [field],
                "confidence": 0.60,
                "reason": "provisional constraint",
            }

        # =================================================
        # FALLBACK
        # =================================================
        return {
            "action": "DEFER",
            "target": "pattern_log",
            "scope": [field],
            "confidence": 0.0,
            "reason": "unclassified signal",
        }

    except Exception as e:
        return {
            "action": "REJECT",
            "target": None,
            "scope": [],
            "confidence": 0.0,
            "reason": f"reasoning_error: {str(e)}",
        }
