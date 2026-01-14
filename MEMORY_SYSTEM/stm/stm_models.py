# stm_models.py
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    latest_version: int

@dataclass
class SessionState:
    current_goal: Optional[str] = None
    stage: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    active_decisions: List[str] = field(default_factory=list)
    artifacts_created: List[ArtifactRef] = field(default_factory=list)
