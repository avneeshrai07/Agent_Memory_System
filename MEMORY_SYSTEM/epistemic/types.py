# MEMORY_SYSTEM.epistemic.types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class RuleCategory(str, Enum):
    INVARIANT = "invariant"
    PRINCIPLE = "principle"
    HEURISTIC = "heuristic"


class RuleScope(str, Enum):
    MEMORY_WRITE = "memory_write"
    MEMORY_RETRIEVAL = "memory_retrieval"
    REASONING = "reasoning"
    GLOBAL = "global"


@dataclass(frozen=True)
class EpistemicRule:
    rule_id: str
    category: RuleCategory
    statement: str
    scope: RuleScope
    priority: int
    overrideable: bool
    rationale: Optional[str]
    introduced_in: str
