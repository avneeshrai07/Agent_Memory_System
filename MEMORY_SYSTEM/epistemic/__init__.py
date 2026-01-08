# MEMORY_SYSTEM/epistemic/__init__.py
from pathlib import Path
from .loader import load_epistemic_rules
from MEMORY_SYSTEM.epistemic.epistemic_engine import EpistemicEngine

_epistemic_engine = None


def get_epistemic_engine() -> EpistemicEngine:
    global _epistemic_engine
    if _epistemic_engine is None:
        rules_path = Path(__file__).parent / "rules_v1.yaml"
        rules = load_epistemic_rules(rules_path)
        _epistemic_engine = EpistemicEngine(rules=rules)
    return _epistemic_engine
