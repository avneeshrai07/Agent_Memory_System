# cognition/cognition_model.py

from typing import Dict, Any


class CognitionModel:
    """
    Agent epistemic configuration.

    - Loaded from DB or config
    - Never queries DB itself
    - Single source of truth for cognition thresholds
    """

    def __init__(self, config: Dict[str, Any]):
        # =====================================================
        # COMMIT THRESHOLDS
        # =====================================================

        self.style_commit_threshold = config.get("style_commit_threshold", 0.65)
        self.identity_commit_threshold = config.get("identity_commit_threshold", 0.80)
        self.organization_commit_threshold = config.get("organization_commit_threshold", 0.90)
        self.constraint_commit_threshold = config.get("constraint_commit_threshold", 0.95)

        # =====================================================
        # REINFORCEMENT REQUIREMENTS
        # =====================================================

        self.implicit_confirmation_required = config.get("implicit_confirmation_required", 2)
        self.organization_confirmation_required = config.get("organization_confirmation_required", 2)

        # =====================================================
        # VOLATILITY MODEL
        # =====================================================

        self.field_volatility = config.get("field_volatility", {})
        self.volatility_penalty = config.get(
            "volatility_penalty",
            {
                "low": 0.05,
                "medium": 0.25,
                "high": 0.40,
            },
        )

        # 🔒 HARD GUARANTEE: style fields are always low volatility
        self._style_fields = {
            "tone",
            "voice",
            "style",
            "length_preference",
            "preferred_format",
            "complexity",
        }

    # -----------------------------------------------------
    # VOLATILITY ACCESSOR (FIXED)
    # -----------------------------------------------------

    def get_volatility_penalty(self, field: str) -> float:
        """
        Return volatility penalty for a given field.

        RULE:
        - STYLE_FIELDS are ALWAYS low volatility
        - Everything else uses config or defaults to high
        """

        if field in self._style_fields:
            return float(self.volatility_penalty.get("low", 0.05))

        volatility_class = self.field_volatility.get(field, "high")
        return float(self.volatility_penalty.get(volatility_class, 0.40))
