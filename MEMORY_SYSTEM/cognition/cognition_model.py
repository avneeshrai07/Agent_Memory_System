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
        # COMMIT THRESHOLDS (CATEGORY-AWARE)
        # =====================================================

        # Style / preference signals (low risk)
        self.style_commit_threshold = config.get(
            "style_commit_threshold", 0.65
        )

        # Identity signals (who the user is)
        self.identity_commit_threshold = config.get(
            "identity_commit_threshold", 0.80
        )

        # Organization context (very stable, high impact)
        self.organization_commit_threshold = config.get(
            "organization_commit_threshold", 0.90
        )

        # Hard constraints (must be explicit)
        self.constraint_commit_threshold = config.get(
            "constraint_commit_threshold", 0.95
        )

        # =====================================================
        # REINFORCEMENT REQUIREMENTS
        # =====================================================

        # For identity-style signals
        self.implicit_confirmation_required = config.get(
            "implicit_confirmation_required", 2
        )

        # For organization signals
        self.organization_confirmation_required = config.get(
            "organization_confirmation_required", 2
        )

        # =====================================================
        # VOLATILITY MODEL
        # =====================================================

        # Field → volatility class (low / medium / high)
        self.field_volatility = config.get(
            "field_volatility", {}
        )

        # Volatility class → numeric penalty
        self.volatility_penalty = config.get(
            "volatility_penalty", {
                "low": 0.10,
                "medium": 0.25,
                "high": 0.40,
            }
        )

    # -----------------------------------------------------
    # VOLATILITY ACCESSOR
    # -----------------------------------------------------

    def get_volatility_penalty(self, field: str) -> float:
        """
        Return volatility penalty for a given field.
        Defaults to 'high' volatility.
        """
        volatility_class = self.field_volatility.get(field, "high")
        return float(self.volatility_penalty.get(volatility_class, 0.40))
