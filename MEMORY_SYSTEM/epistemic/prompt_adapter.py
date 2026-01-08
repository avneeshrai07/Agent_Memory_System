from typing import List
from MEMORY_SYSTEM.epistemic.types import EpistemicRule, RuleScope, RuleCategory


def build_epistemic_prompt_block(
    rules: List[EpistemicRule],
    scope: RuleScope
) -> str:
    """
    Convert epistemic rules into an LLM-safe prompt block.
    Fail-safe: returns empty string on error.
    """
    try:
        relevant_rules = [
            r for r in rules
            if r.scope in (scope, RuleScope.GLOBAL)
        ]

        invariant_lines = []
        principle_lines = []

        for rule in relevant_rules:
            if rule.category == RuleCategory.INVARIANT:
                invariant_lines.append(f"- MUST: {rule.statement}")
            elif rule.category == RuleCategory.PRINCIPLE:
                principle_lines.append(f"- SHOULD: {rule.statement}")

        blocks = []

        if invariant_lines:
            blocks.append(
                "NON-NEGOTIABLE CONSTRAINTS:\n" +
                "\n".join(invariant_lines)
            )

        if principle_lines:
            blocks.append(
                "DEFAULT REASONING RULES:\n" +
                "\n".join(principle_lines)
            )

        return "\n\n".join(blocks)

    except Exception as e:
        # Never break the LLM call because of epistemic formatting
        return ""
