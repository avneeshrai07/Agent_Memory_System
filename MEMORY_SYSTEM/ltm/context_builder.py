from typing import List, Dict
import traceback


def build_ltm_context(memories: List[Dict]) -> str:
    """
    Convert retrieved LTM facts into a prompt-ready context block.

    - Returns ONLY context (never user query)
    - Never throws
    - Safe for direct prompt injection
    """

    try:
        if not memories:
            return ""

        lines = []
        for mem in memories:
            fact = mem.get("fact")
            if not fact:
                continue
            lines.append(f"- {fact}")

        if not lines:
            return ""

        context = (
            "Relevant known context from prior interactions:\n"
            + "\n".join(lines)
        )

        return context

    except Exception:
        print("❌ [LTM-CONTEXT] Failed to build context")
        traceback.print_exc()
        return ""
