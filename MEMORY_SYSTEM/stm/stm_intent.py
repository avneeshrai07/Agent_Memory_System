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
