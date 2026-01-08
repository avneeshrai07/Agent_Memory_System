# cognition/reasoning_policy.py

from MEMORY_SYSTEM.cognition.decision_schema import CognitionDecision
from MEMORY_SYSTEM.cognition.cognition_model import CognitionModel
from typing import Dict, Any

# ---------------------------------------------------------------------
# Explicit intent detection
# ---------------------------------------------------------------------

# Linguistic markers that imply permanence / explicit preference
EXPLICIT_MARKERS = {
    "always",
    "never",
    "not a one-time",
    "not one-time",
    "permanent",
    "must",
    "do not",
    "avoid",
}

# Fields safe to commit on first explicit signal
LOW_RISK_FIELDS = {
    "tone",
    "writing_style",
    "content_format",
    "language",
}


def is_explicit_equivalent(signal: Dict[str, Any]) -> bool:
    """
    Treat derived signals as explicit if they contain strong
    permanence language.
    """
    value = str(signal.get("value", "")).lower()
    return any(marker in value for marker in EXPLICIT_MARKERS)


# ---------------------------------------------------------------------
# Core decision function
# ---------------------------------------------------------------------

async def decide(
    signal: Dict[str, Any],
    cognition_model: CognitionModel,
) -> Dict[str, Any]:
    """
    Produce a CognitionDecision for a single signal.
    This function has NO side effects.
    """

    try:
        field = signal.get("field")
        category = signal.get("category")
        source = signal.get("source", "derived")
        frequency = int(signal.get("frequency", 1))

        base_confidence = float(signal.get("base_confidence", 0.0))

        # -------------------------------------------------
        # Volatility penalty
        # -------------------------------------------------
        volatility_penalty = cognition_model.get_volatility_penalty(field)
        final_confidence = max(base_confidence - volatility_penalty, 0.0)

        # -------------------------------------------------
        # Explicit / explicit-equivalent detection
        # -------------------------------------------------
        explicit_like = (
            source == "explicit"
            or is_explicit_equivalent(signal)
        )

        # -------------------------------------------------
        # RULE 1: Explicit preference for LOW-RISK fields
        # -------------------------------------------------
        if (
            explicit_like
            and field in LOW_RISK_FIELDS
            and final_confidence >= 0.7
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "explicit preference for low-risk field",
            }

        # -------------------------------------------------
        # RULE 2: Reinforced implicit / derived signals
        # -------------------------------------------------
        if (
            source != "explicit"
            and frequency >= cognition_model.implicit_confirmation_required
            and final_confidence >= cognition_model.explicit_commit_threshold
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "reinforced signal reached confirmation threshold",
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
        # Cognition must never crash the pipeline
        return {
            "action": "REJECT",
            "target": None,
            "scope": [],
            "confidence": 0.0,
            "reason": f"reasoning_error: {str(e)}",
        }