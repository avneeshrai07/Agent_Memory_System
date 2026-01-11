from copy import deepcopy
from typing import List, Dict, Optional

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel


# =========================================================
# SIGNAL EMISSION (FIELD-LEVEL, EXPLICIT PERSONA ONLY)
# =========================================================

from typing import List, Dict, Optional
from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel


def persona_to_signals(
    extracted_persona: Optional[UserPersonaModel],
) -> List[Dict]:
    """
    Convert structured persona blocks into persona-authoritative signals.

    HARD RULES:
    - These signals represent explicit user-declared persona
    - They are NOT inferred
    - They are NOT probabilistic
    - They MUST bypass learning cognition
    """

    # 🔒 HARD GUARD — extraction is probabilistic
    if extracted_persona is None:
        return []

    signals: List[Dict] = []

    def add_signal(
        *,
        category: str,
        field: str,
        value,
        confidence: Optional[float],
    ) -> None:
        if value is None:
            return
        if confidence is None or confidence <= 0:
            return

        signals.append({
            "category": category,          # identity | organization | preference | constraint
            "field": field,
            "value": value,

            # epistemics
            "base_confidence": confidence,
            "source": "explicit",
            "epistemic_role": "persona",
            "frequency": 1,
        })

    # =====================================================
    # USER IDENTITY
    # =====================================================
    ui = extracted_persona.user_identity
    if ui:
        add_signal(category="identity", field="job_title", value=ui.job_title, confidence=ui.confidence)
        add_signal(category="identity", field="seniority", value=ui.seniority, confidence=ui.confidence)
        add_signal(category="identity", field="function", value=ui.function, confidence=ui.confidence)
        add_signal(category="identity", field="decision_authority", value=ui.decision_authority, confidence=ui.confidence)

    # =====================================================
    # COMPANY — PROFILE
    # =====================================================
    cp = extracted_persona.company_profile
    if cp:
        add_signal(category="organization", field="company_name", value=cp.company_name, confidence=cp.confidence)
        add_signal(category="organization", field="industry", value=cp.industry, confidence=cp.confidence)
        add_signal(category="organization", field="company_size", value=cp.company_size, confidence=cp.confidence)
        add_signal(category="organization", field="company_stage", value=cp.company_stage, confidence=cp.confidence)

    # =====================================================
    # COMPANY — BUSINESS
    # =====================================================
    cb = extracted_persona.company_business
    if cb:
        add_signal(category="organization", field="business_model", value=cb.business_model, confidence=cb.confidence)
        add_signal(category="organization", field="sales_motion", value=cb.sales_motion, confidence=cb.confidence)
        add_signal(category="organization", field="target_customers", value=cb.target_customers, confidence=cb.confidence)

    # =====================================================
    # COMPANY — PRODUCTS / TECH
    # =====================================================
    cpb = extracted_persona.company_products
    if cpb:
        add_signal(category="organization", field="products", value=cpb.products, confidence=cpb.confidence)
        add_signal(category="organization", field="tech_orientation", value=cpb.tech_orientation, confidence=cpb.confidence)

    # =====================================================
    # OBJECTIVE
    # =====================================================
    obj = extracted_persona.objective
    if obj:
        add_signal(category="preference", field="primary_goal", value=obj.primary_goal, confidence=obj.confidence)
        add_signal(category="preference", field="desired_action", value=obj.desired_action, confidence=obj.confidence)
        add_signal(category="preference", field="success_criteria", value=obj.success_criteria, confidence=obj.confidence)

    # =====================================================
    # CONTENT FORMAT
    # =====================================================
    cf = extracted_persona.content_format
    if cf:
        add_signal(category="preference", field="preferred_format", value=cf.preferred_format, confidence=cf.confidence)
        add_signal(category="preference", field="length_preference", value=cf.length_preference, confidence=cf.confidence)

    # =====================================================
    # AUDIENCE
    # =====================================================
    aud = extracted_persona.audience
    if aud:
        add_signal(category="preference", field="audience_type", value=aud.audience_type, confidence=aud.confidence)
        add_signal(category="preference", field="audience_domain", value=aud.audience_domain, confidence=aud.confidence)
        add_signal(category="preference", field="audience_level", value=aud.audience_level, confidence=aud.confidence)

    # =====================================================
    # TONE
    # =====================================================
    tone = extracted_persona.tone
    if tone:
        add_signal(category="preference", field="tone", value=tone.tone, confidence=tone.confidence)
        add_signal(category="preference", field="voice", value=tone.voice, confidence=tone.confidence)
        add_signal(category="preference", field="emotional_intensity", value=tone.emotional_intensity, confidence=tone.confidence)

    # =====================================================
    # WRITING STYLE
    # =====================================================
    ws = extracted_persona.writing_style
    if ws:
        add_signal(category="preference", field="style", value=ws.style, confidence=ws.confidence)
        add_signal(category="preference", field="sentence_structure", value=ws.sentence_structure, confidence=ws.confidence)

    # =====================================================
    # LANGUAGE
    # =====================================================
    lang = extracted_persona.language
    if lang:
        add_signal(category="preference", field="language", value=lang.language, confidence=lang.confidence)
        add_signal(category="preference", field="complexity", value=lang.complexity, confidence=lang.confidence)
        add_signal(category="preference", field="jargon_policy", value=lang.jargon_policy, confidence=lang.confidence)

    # =====================================================
    # CONSTRAINTS
    # =====================================================
    cons = extracted_persona.constraints
    if cons:
        add_signal(category="constraint", field="constraints", value=cons.constraints, confidence=cons.confidence)

    return signals


# =========================================================
# FIELD → PERSONA BLOCK MAP (AUTHORITATIVE)
# =========================================================

FIELD_TO_BLOCK = {
    # identity
    "job_title": ("user_identity", "job_title"),
    "seniority": ("user_identity", "seniority"),
    "function": ("user_identity", "function"),
    "decision_authority": ("user_identity", "decision_authority"),

    # company profile
    "company_name": ("company_profile", "company_name"),
    "industry": ("company_profile", "industry"),
    "company_size": ("company_profile", "company_size"),
    "company_stage": ("company_profile", "company_stage"),

    # company business
    "business_model": ("company_business", "business_model"),
    "sales_motion": ("company_business", "sales_motion"),
    "target_customers": ("company_business", "target_customers"),

    # company products
    "products": ("company_products", "products"),
    "tech_orientation": ("company_products", "tech_orientation"),

    # objective
    "primary_goal": ("objective", "primary_goal"),
    "desired_action": ("objective", "desired_action"),
    "success_criteria": ("objective", "success_criteria"),

    # content format
    "preferred_format": ("content_format", "preferred_format"),
    "length_preference": ("content_format", "length_preference"),

    # audience
    "audience_type": ("audience", "audience_type"),
    "audience_domain": ("audience", "audience_domain"),
    "audience_level": ("audience", "audience_level"),

    # tone / style
    "tone": ("tone", "tone"),
    "voice": ("tone", "voice"),
    "emotional_intensity": ("tone", "emotional_intensity"),
    "style": ("writing_style", "style"),
    "sentence_structure": ("writing_style", "sentence_structure"),

    # language
    "language": ("language", "language"),
    "complexity": ("language", "complexity"),
    "jargon_policy": ("language", "jargon_policy"),

    # constraints
    "constraints": ("constraints", "constraints"),
}


# =========================================================
# PERSONA PROJECTION (DECISION-DRIVEN, SAFE)
# =========================================================

def project_persona_by_decisions(
    extracted_persona: UserPersonaModel,
    decisions: List[Dict],
) -> Optional[UserPersonaModel]:
    """
    Project a partial persona based strictly on persona COMMIT decisions.

    Guarantees:
    - Field-accurate
    - No silent carryover
    - No learning metadata leakage
    """

    projected_blocks: Dict[str, Dict] = {}
    any_commit = False

    for decision in decisions:
        if decision.get("action") not in {"COMMIT", "PARTIAL_COMMIT"}:
            continue
        if decision.get("target") != "persona":
            continue

        scope = decision.get("scope")
        if not scope:
            # Defensive: malformed cognition output
            continue

        for field in scope:
            mapping = FIELD_TO_BLOCK.get(field)
            if not mapping:
                continue

            block_name, attr_name = mapping
            source_block = getattr(extracted_persona, block_name, None)
            if source_block is None:
                continue

            value = getattr(source_block, attr_name, None)
            if value is None:
                continue

            if block_name not in projected_blocks:
                projected_blocks[block_name] = {}

            projected_blocks[block_name][attr_name] = value
            any_commit = True

    if not any_commit:
        return None

    # Build persona strictly from committed fields
    return UserPersonaModel(**projected_blocks)
