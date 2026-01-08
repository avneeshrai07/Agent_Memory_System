# cognition/reasoning_policy.py

from typing import Dict, Any
from MEMORY_SYSTEM.cognition.cognition_model import CognitionModel

# ---------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------

STYLE_FIELDS = {
    "tone",
    "writing_style",
    "content_format",
    "language",
    "constraints",
}

IDENTITY_FIELDS = {
    "objective",
    "audience",
}

# ---------------------------------------------------------------------
# Core decision function
# ---------------------------------------------------------------------

async def decide(
    signal: Dict[str, Any],
    cognition_model: CognitionModel,
) -> Dict[str, Any]:
    """
    Produce a CognitionDecision for a single signal.
    No side effects. Pure policy.
    """

    try:
        field = signal.get("field")
        source = signal.get("source", "derived")
        frequency = int(signal.get("frequency", 1))
        base_confidence = float(signal.get("base_confidence", 0.0))

        # -------------------------------------------------
        # Volatility penalty (decays with frequency)
        # -------------------------------------------------
        volatility_penalty = cognition_model.get_volatility_penalty(field)
        effective_volatility = volatility_penalty / max(frequency, 1)
        final_confidence = max(base_confidence - effective_volatility, 0.0)

        # -------------------------------------------------
        # RULE 1: STYLE preferences → commit by default
        # -------------------------------------------------
        if (
            field in STYLE_FIELDS
            and final_confidence >= 0.65
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "stylistic preference assumed persistent",
            }

        # -------------------------------------------------
        # RULE 2: IDENTITY preferences → require reinforcement
        # -------------------------------------------------
        if (
            field in IDENTITY_FIELDS
            and frequency >= cognition_model.implicit_confirmation_required
            and final_confidence >= cognition_model.explicit_commit_threshold
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "reinforced identity preference",
            }

        # -------------------------------------------------
        # RULE 3: Default → defer
        # -------------------------------------------------
        return {
            "action": "DEFER",
            "target": "pattern_log",
            "scope": [field],
            "confidence": round(final_confidence, 2),
            "reason": "not enough confidence",
        }

    except Exception as e:
        return {
            "action": "REJECT",
            "target": None,
            "scope": [],
            "confidence": 0.0,
            "reason": f"reasoning_error: {str(e)}",
        }
