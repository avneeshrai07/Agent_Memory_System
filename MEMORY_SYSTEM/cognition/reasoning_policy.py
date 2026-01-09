# cognition/reasoning_policy.py

from typing import Dict, Any
from MEMORY_SYSTEM.cognition.cognition_model import CognitionModel

# ---------------------------------------------------------------------
# Field classification (SEMANTIC, NOT STRUCTURAL)
# ---------------------------------------------------------------------

# Low-risk, high-volatility fields
STYLE_FIELDS = {
    "tone",
    "voice",
    "style",
    "length_preference",
    "preferred_format",
    "complexity",
}

# Stable but identity-defining
IDENTITY_FIELDS = {
    "job_title",
    "seniority",
    "function",
    "decision_authority",
}

# Organizational context (slow changing, high impact)
ORGANIZATION_FIELDS = {
    "company_name",
    "industry",
    "company_size",
    "company_stage",
    "business_model",
    "sales_motion",
    "target_customers",
    "products",
    "tech_orientation",
}

# Hard constraints (must be explicit)
CONSTRAINT_FIELDS = {
    "constraints",
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

    Guarantees:
    - No side effects
    - No inference
    - One signal → one decision
    """

    try:
        field = signal.get("field")
        category = signal.get("category")
        frequency = int(signal.get("frequency", 1))
        base_confidence = float(signal.get("base_confidence", 0.0))

        # -------------------------------------------------
        # Volatility penalty (decays with reinforcement)
        # -------------------------------------------------
        volatility_penalty = cognition_model.get_volatility_penalty(field)
        effective_volatility = volatility_penalty / max(frequency, 1)
        final_confidence = max(base_confidence - effective_volatility, 0.0)

        # -------------------------------------------------
        # RULE 1: STYLE / PREFERENCE → easy partial commit
        # -------------------------------------------------
        if (
            category == "preference"
            and field in STYLE_FIELDS
            and final_confidence >= cognition_model.style_commit_threshold
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "stylistic preference with sufficient confidence",
            }
        
        # -------------------------------------------------
        # RULE 1.5: PROVISIONAL MEMORY (UX-SAFE ACCEPTANCE)
        # -------------------------------------------------

        if (
            category in {"identity", "organization", "preference", "constraint"}
            and frequency == 1
            and base_confidence >= 0.95
        ):
            return {
                "action": "PROVISIONAL_COMMIT",
                "target": "runtime_only",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "accepted provisionally for working context",
            }


        # -------------------------------------------------
        # RULE 2: IDENTITY → require reinforcement
        # -------------------------------------------------
        if (
            category == "identity"
            and field in IDENTITY_FIELDS
            and frequency >= cognition_model.implicit_confirmation_required
            and final_confidence >= cognition_model.identity_commit_threshold
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "reinforced identity signal",
            }

        # -------------------------------------------------
        # RULE 3: ORGANIZATION → very conservative
        # -------------------------------------------------
        if (
            category == "organization"
            and field in ORGANIZATION_FIELDS
            and frequency >= cognition_model.organization_confirmation_required
            and final_confidence >= cognition_model.organization_commit_threshold
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "stable organizational context confirmed",
            }

        # -------------------------------------------------
        # RULE 4: CONSTRAINTS → explicit only
        # -------------------------------------------------
        if (
            category == "constraint"
            and field in CONSTRAINT_FIELDS
            and final_confidence >= cognition_model.constraint_commit_threshold
        ):
            return {
                "action": "PARTIAL_COMMIT",
                "target": "persona",
                "scope": [field],
                "confidence": round(final_confidence, 2),
                "reason": "explicit constraint accepted",
            }

        # -------------------------------------------------
        # RULE 5: Default → defer
        # -------------------------------------------------
        return {
            "action": "DEFER",
            "target": "pattern_log",
            "scope": [field],
            "confidence": round(final_confidence, 2),
            "reason": "insufficient confidence or reinforcement",
        }

    except Exception as e:
        return {
            "action": "REJECT",
            "target": None,
            "scope": [],
            "confidence": 0.0,
            "reason": f"reasoning_error: {str(e)}",
        }
