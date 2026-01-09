from typing import Dict, Any, Optional

# ============================================================
# FIELD POLICY DEFINITION
# ============================================================

FIELD_POLICY = {

    # ---------------- IDENTITY ----------------
    "full_name":          {"mode": "explicit", "persona": True},
    "preferred_name":     {"mode": "explicit", "persona": True},
    "job_title":          {"mode": "implicit", "min_freq": 2},
    "seniority":          {"mode": "hybrid",   "min_freq": 2},
    "function":           {"mode": "implicit", "min_freq": 2},
    "decision_authority": {"mode": "hybrid",   "min_freq": 2},
    "years_experience":   {"mode": "explicit", "persona": True},

    # ---------------- COMPANY ----------------
    "company_name":   {"mode": "implicit", "min_freq": 2},
    "registered_name":{"mode": "explicit"},
    "industry":       {"mode": "implicit", "min_freq": 2},
    "company_stage":  {"mode": "hybrid",   "min_freq": 2},
    "company_size":   {"mode": "hybrid",   "min_freq": 2},
    "headquarters":   {"mode": "explicit"},
    "geo_market":     {"mode": "explicit"},
    "website":        {"mode": "explicit"},

    # ---------------- BUSINESS ----------------
    "business_model":         {"mode": "implicit", "min_freq": 2},
    "pricing_model":          {"mode": "explicit"},
    "sales_motion":           {"mode": "hybrid",   "min_freq": 2},
    "core_value_proposition": {"mode": "explicit_or_n", "min_freq": 3},

    # ---------------- PRODUCT ----------------
    "products":             {"mode": "explicit"},
    "tech_orientation":     {"mode": "hybrid",   "min_freq": 2},
    "differentiation_axes": {"mode": "explicit_or_n", "min_freq": 3},

    # ---------------- BRAND ----------------
    "core_values":               {"mode": "explicit_or_n", "min_freq": 3},
    "brand_personality":         {"mode": "explicit_or_n", "min_freq": 3},
    "compliance_sensitivity":    {"mode": "explicit"},
    "data_security_sensitivity": {"mode": "explicit"},

    # ---------------- OBJECTIVES (TASK ONLY) ----------------
    "primary_goal":     {"mode": "explicit", "persona": False},
    "desired_action":   {"mode": "explicit", "persona": False},
    "success_criteria": {"mode": "explicit", "persona": False},
    "horizon":          {"mode": "explicit", "persona": False},

    # ---------------- CONTENT FORMAT (TASK ONLY) ----------------
    "content_types":     {"mode": "explicit", "persona": False},
    "preferred_format":  {"mode": "explicit", "persona": False},
    "length_preference": {"mode": "implicit", "min_freq": 2},

    # ---------------- AUDIENCE ----------------
    "audience_type":   {"mode": "implicit", "min_freq": 2},
    "audience_domain": {"mode": "implicit", "min_freq": 2},
    "audience_level":  {"mode": "implicit", "min_freq": 2},
    "geo_context":     {"mode": "explicit"},

    # ---------------- TONE ----------------
    "tone":                {"mode": "hybrid", "min_freq": 2},
    "voice":               {"mode": "implicit", "min_freq": 2},
    "emotional_intensity": {"mode": "explicit_or_n", "min_freq": 3},

    # ---------------- WRITING STYLE ----------------
    "style":              {"mode": "explicit_or_n", "min_freq": 3},
    "sentence_structure": {"mode": "implicit", "min_freq": 2},
    "use_examples":       {"mode": "hybrid", "min_freq": 2},
    "use_storytelling":   {"mode": "explicit_or_n", "min_freq": 3},

    # ---------------- LANGUAGE ----------------
    "language":       {"mode": "implicit", "min_freq": 2},
    "complexity":     {"mode": "implicit", "min_freq": 2},
    "jargon_policy":  {"mode": "hybrid",   "min_freq": 2},

    # ---------------- CONSTRAINTS ----------------
    "constraints": {
        "mode": "explicit_or_n",
        "min_freq": 2,
        "min_confidence": 0.95,   # HARD SAFETY RULE
    },
}

# ============================================================
# ACTION HELPERS (STRICT & EXPLICIT)
# ============================================================

def _commit(target: str, reason: str) -> Dict[str, Any]:
    return {
        "action": "COMMIT",
        "target": target,     # "persona" | "runtime"
        "confidence": 1.0,
        "reason": reason,
    }

def _provisional(reason: str) -> Dict[str, Any]:
    return {
        "action": "PROVISIONAL_COMMIT",
        "target": "runtime",
        "confidence": 0.6,
        "reason": reason,
    }

def _reject(reason: str) -> Dict[str, Any]:
    return {
        "action": "REJECT",
        "target": None,
        "confidence": 0.0,
        "reason": reason,
    }

def _defer(reason: str) -> Dict[str, Any]:
    return {
        "action": "DEFER",
        "target": "pattern_log",
        "confidence": 0.0,
        "reason": reason,
    }

# ============================================================
# MAIN REASONING POLICY
# ============================================================

async def decide(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Production-grade cognition decision engine.

    Guarantees:
    - No persona pollution
    - No format locking
    - No implicit identity hallucination
    - Objectives remain task-scoped
    - Constraints are safety-gated
    """

    try:
        field: Optional[str] = signal.get("field")
        frequency: int = int(signal.get("frequency", 1))
        confidence: float = float(signal.get("base_confidence", 0.0))
        explicit: bool = bool(signal.get("explicit", False))

        if not field:
            return _defer("missing field")

        policy = FIELD_POLICY.get(field)
        if not policy:
            return _defer("unknown field")

        # ---------------- Confidence Gate ----------------
        min_conf = policy.get("min_confidence", 0.80)
        if confidence < min_conf:
            return _reject("confidence below threshold")

        mode = policy["mode"]
        min_freq = policy.get("min_freq", 0)
        persona_allowed = policy.get("persona", True)
        target = "persona" if persona_allowed else "runtime"

        # ---------------- Explicit Only ----------------
        if mode == "explicit":
            if not explicit:
                return _reject("explicit-only field")
            return _commit(target, "explicit statement")

        # ---------------- Explicit OR N-times ----------------
        if mode == "explicit_or_n":
            if explicit:
                return _commit(target, "explicit statement")
            if frequency >= min_freq:
                return _commit(target, "implicit repetition threshold met")
            return _provisional("awaiting repetition")

        # ---------------- Implicit ----------------
        if mode == "implicit":
            if frequency >= min_freq:
                return _commit(target, "implicit repetition threshold met")
            return _provisional("insufficient repetition")

        # ---------------- Hybrid ----------------
        if mode == "hybrid":
            if explicit:
                return _commit(target, "explicit hybrid signal")
            if frequency >= min_freq:
                return _commit(target, "implicit hybrid repetition")
            return _provisional("hybrid awaiting confirmation")

        return _defer("unhandled policy mode")

    except Exception as e:
        return _reject(f"reasoning_error: {str(e)}")
