# intent_detector.py
import re
from MEMORY_SYSTEM.stm.intent_detector import Intent
from stm_models import SessionState

EDIT_VERBS = ["change", "edit", "update", "modify", "tweak"]
REWRITE_VERBS = ["rewrite", "redo", "regenerate"]
SMALL_SCOPE = ["slightly", "small", "just", "soften"]

def detect_intent(message: str, stm: SessionState) -> Intent:
    msg = message.lower()

    day_match = re.search(r"day\s*(\d+)", msg)
    artifact_ref = None

    if day_match:
        day = day_match.group(1)
        for a in stm.artifacts_created:
            if f"day:{day}" in a.artifact_id:
                artifact_ref = a.artifact_id

    if any(v in msg for v in EDIT_VERBS) and artifact_ref:
        scope = "small_change" if any(s in msg for s in SMALL_SCOPE) else "global_change"
        return Intent(
            name="artifact_edit",
            artifact_id=artifact_ref,
            edit_scope=scope
        )

    if any(v in msg for v in REWRITE_VERBS) and artifact_ref:
        return Intent(
            name="artifact_regenerate",
            artifact_id=artifact_ref
        )

    if "write" in msg or "create" in msg:
        return Intent(name="content_generate")

    return Intent(name="conversation_continue")
