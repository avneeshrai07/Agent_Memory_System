# MEMORY_SYSTEM/storage/stm_record.py

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

STMType = Literal[
    "goal",
    "decision",
    "constraint",
    "approval",
    "rejection",
    "direction_change",
    "scope",
]


@dataclass
class STMRecord:
    user_id: str
    state_type: STMType
    statement: str
    rationale: str | None
    confidence: float | None
    created_at: datetime
    is_active: bool = True
