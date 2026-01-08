# cognition/cognition_model.py

from typing import Dict, Any


class CognitionModel:
    """
    Agent epistemic configuration.
    Loaded FROM DB, but never queries DB itself.
    """

    def __init__(self, config: Dict[str, Any]):
        self.explicit_commit_threshold = config.get(
            "explicit_commit_threshold", 0.85
        )
        self.implicit_confirmation_required = config.get(
            "implicit_confirmation_required", 2
        )

        self.field_volatility = config.get("field_volatility", {})
        self.volatility_penalty = config.get("volatility_penalty", {})

    def get_volatility_penalty(self, field: str) -> float:
        volatility = self.field_volatility.get(field, "high")
        return self.volatility_penalty.get(volatility, 0.30)
