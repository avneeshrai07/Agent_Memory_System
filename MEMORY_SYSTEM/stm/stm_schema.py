from pydantic import BaseModel
from typing import Optional, Literal
from uuid import UUID
from datetime import datetime


class STMEntry(BaseModel):
    stm_id: UUID
    timestamp: datetime

    state_type: Literal[
        "goal",
        "decision",
        "constraint",
        "approval",
        "rejection",
        "direction_change",
        "scope"
    ]

    statement: str
    rationale: Optional[str] = None

    applies_to: Optional[UUID] = None
    supersedes: Optional[UUID] = None

    confidence: float = 1.0
