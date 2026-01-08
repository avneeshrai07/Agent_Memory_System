from MEMORY_SYSTEM.epistemic import get_epistemic_engine
from MEMORY_SYSTEM.epistemic.types import RuleScope
from MEMORY_SYSTEM.epistemic.prompt_adapter import build_epistemic_prompt_block


def build_epistemic_system_prompt(base_prompt: str) -> str:
    try:
        engine = get_epistemic_engine()

        epistemic_block = build_epistemic_prompt_block(
            rules=engine.rules,
            scope=RuleScope.REASONING
        )

        if epistemic_block.strip():
            return f"""
{epistemic_block}

----------------------------
AGENT ROLE INSTRUCTIONS:
{base_prompt}
""".strip()

        return base_prompt.strip()

    except Exception:
        return base_prompt.strip()
