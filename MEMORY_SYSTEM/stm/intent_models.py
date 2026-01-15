# intent_models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Intent:
    name: str
    artifact_id: Optional[str] = None
    edit_scope: Optional[str] = None
