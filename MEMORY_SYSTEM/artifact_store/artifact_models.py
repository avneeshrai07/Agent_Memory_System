# artifact_models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
@dataclass
class Artifact:
    artifact_id: str
    artifact_type: str
    version: int
    content: str
    created_at: datetime
    change_reason: Optional[str] = None
