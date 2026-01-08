# cognition/decision_schema.py

from typing import List, Optional, Literal, Dict, Any

ActionType = Literal[
    "COMMIT",
    "PARTIAL_COMMIT",
    "DEFER",
    "REJECT",
]

TargetType = Literal[
    "persona",
    "evidence_ltm",
    "pattern_log",
    "runtime_only",
]

class CognitionDecision(Dict[str, Any]):
    """
    Canonical decision object produced by Cognition.
    """

    action: ActionType
    target: Optional[TargetType]

    # which fields or facts are affected
    scope: List[str]

    # final confidence after cognition math
    confidence: float

    # human-readable explanation (for logs / debugging)
    reason: str
