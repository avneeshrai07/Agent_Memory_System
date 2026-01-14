from pydantic import BaseModel
from typing import Optional, Literal


class STMIntent(BaseModel):
    should_write: bool

    state_type: Optional[Literal[
        "goal",
        "decision",
        "constraint",
        "approval",
        "rejection",
        "direction_change",
        "scope"
    ]] = None

    statement: Optional[str] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None


class RouteIntent(BaseModel):
    route: Literal[
        "current_context",
        "edit",
        "reference",
        "semantic_lookup"
    ]
    confidence: float


class CombinedIntent(BaseModel):
    stm: STMIntent
    route: RouteIntent