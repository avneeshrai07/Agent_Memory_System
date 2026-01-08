# MEMORY_SYSTEM.epistemic.engine.py
from typing import Iterable
from MEMORY_SYSTEM.epistemic.types import EpistemicRule, RuleScope


class EpistemicEngine:
    def __init__(self, rules: Iterable[EpistemicRule]):
        self.rules = list(rules)

    def rules_for_scope(self, scope: RuleScope) -> list[EpistemicRule]:
        return [r for r in self.rules if r.scope in (scope, RuleScope.GLOBAL)]

    def assert_allowed(self, scope: RuleScope, context: dict):
        """
        Called before critical actions.
        Raises exception if an invariant is violated.
        """
        for rule in self.rules_for_scope(scope):
            if rule.category == "invariant":
                self._enforce(rule, context)

    def _enforce(self, rule: EpistemicRule, context: dict):
        """
        Enforcement logic is explicit, not LLM-based.
        """
        # Example enforcement placeholder
        if rule.rule_id == "EPI-001":
            if context.get("overwrite_attempt"):
                raise RuntimeError(
                    f"Epistemic violation [{rule.rule_id}]: {rule.statement}"
                )
