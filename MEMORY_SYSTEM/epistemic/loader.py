# MEMORY_SYSTEM.epistemic.loader.py
import yaml
from pathlib import Path
from MEMORY_SYSTEM.epistemic.types import EpistemicRule, RuleCategory, RuleScope


def load_epistemic_rules(path: Path) -> list[EpistemicRule]:
    raw = yaml.safe_load(path.read_text())

    rules = []
    for r in raw["rules"]:
        rule = EpistemicRule(
            rule_id=r["rule_id"],
            category=RuleCategory(r["category"]),
            scope=RuleScope(r["scope"]),
            priority=int(r["priority"]),
            overrideable=bool(r["overrideable"]),
            statement=r["statement"].strip(),
            rationale=r.get("rationale"),
            introduced_in=r["introduced_in"],
        )
        rules.append(rule)

    # enforce deterministic ordering
    return sorted(rules, key=lambda r: r.priority)
