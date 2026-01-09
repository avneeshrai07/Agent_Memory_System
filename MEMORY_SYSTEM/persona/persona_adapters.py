from copy import deepcopy
from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel


# =========================================================
# SIGNAL EMISSION (FIELD-LEVEL, EXPLICIT ONLY)
# =========================================================

def persona_to_signals(extracted_persona: UserPersonaModel) -> list[dict]:
    """
    Convert structured persona blocks into cognition signal candidates.

    RULES:
    - Only emit signals for explicitly present values
    - Never emit inferred or placeholder signals
    - Confidence is taken directly from persona blocks
    """

    signals: list[dict] = []

    def add_signal(
        *,
        category: str,
        field: str,
        value,
        confidence: float,
    ):
        if value is None:
            return
        if confidence is None or confidence <= 0:
            return

        signals.append({
            "category": category,          # identity | organization | preference | constraint
            "field": field,
            "value": value,
            "base_confidence": confidence,
            "source": "extracted",
            "frequency": 1,
        })

    # =====================================================
    # USER IDENTITY
    # =====================================================
    if extracted_persona.user_identity:
        ui = extracted_persona.user_identity

        add_signal(category="identity", field="job_title", value=ui.job_title, confidence=ui.confidence)
        add_signal(category="identity", field="seniority", value=ui.seniority, confidence=ui.confidence)
        add_signal(category="identity", field="function", value=ui.function, confidence=ui.confidence)
        add_signal(category="identity", field="decision_authority", value=ui.decision_authority, confidence=ui.confidence)

    # =====================================================
    # COMPANY — PROFILE
    # =====================================================
    if extracted_persona.company_profile:
        cp = extracted_persona.company_profile

        add_signal(category="organization", field="company_name", value=cp.company_name, confidence=cp.confidence)
        add_signal(category="organization", field="industry", value=cp.industry, confidence=cp.confidence)
        add_signal(category="organization", field="company_size", value=cp.company_size, confidence=cp.confidence)
        add_signal(category="organization", field="company_stage", value=cp.company_stage, confidence=cp.confidence)

    # =====================================================
    # COMPANY — BUSINESS
    # =====================================================
    if extracted_persona.company_business:
        cb = extracted_persona.company_business

        add_signal(category="organization", field="business_model", value=cb.business_model, confidence=cb.confidence)
        add_signal(category="organization", field="sales_motion", value=cb.sales_motion, confidence=cb.confidence)
        add_signal(category="organization", field="target_customers", value=cb.target_customers, confidence=cb.confidence)

    # =====================================================
    # COMPANY — PRODUCTS / TECH
    # =====================================================
    if extracted_persona.company_products:
        cprod = extracted_persona.company_products

        add_signal(category="organization", field="products", value=cprod.products, confidence=cprod.confidence)
        add_signal(category="organization", field="tech_orientation", value=cprod.tech_orientation, confidence=cprod.confidence)

    # =====================================================
    # OBJECTIVE
    # =====================================================
    if extracted_persona.objective:
        obj = extracted_persona.objective

        add_signal(category="preference", field="primary_goal", value=obj.primary_goal, confidence=obj.confidence)
        add_signal(category="preference", field="desired_action", value=obj.desired_action, confidence=obj.confidence)
        add_signal(category="preference", field="success_criteria", value=obj.success_criteria, confidence=obj.confidence)

    # =====================================================
    # CONTENT FORMAT
    # =====================================================
    if extracted_persona.content_format:
        cf = extracted_persona.content_format

        add_signal(category="preference", field="preferred_format", value=cf.preferred_format, confidence=cf.confidence)
        add_signal(category="preference", field="length_preference", value=cf.length_preference, confidence=cf.confidence)

    # =====================================================
    # AUDIENCE
    # =====================================================
    if extracted_persona.audience:
        aud = extracted_persona.audience

        add_signal(category="preference", field="audience_type", value=aud.audience_type, confidence=aud.confidence)
        add_signal(category="preference", field="audience_domain", value=aud.audience_domain, confidence=aud.confidence)
        add_signal(category="preference", field="audience_level", value=aud.audience_level, confidence=aud.confidence)

    # =====================================================
    # TONE
    # =====================================================
    if extracted_persona.tone:
        tone = extracted_persona.tone

        add_signal(category="preference", field="tone", value=tone.tone, confidence=tone.confidence)
        add_signal(category="preference", field="voice", value=tone.voice, confidence=tone.confidence)
        add_signal(category="preference", field="emotional_intensity", value=tone.emotional_intensity, confidence=tone.confidence)

    # =====================================================
    # WRITING STYLE
    # =====================================================
    if extracted_persona.writing_style:
        ws = extracted_persona.writing_style

        add_signal(category="preference", field="style", value=ws.style, confidence=ws.confidence)
        add_signal(category="preference", field="sentence_structure", value=ws.sentence_structure, confidence=ws.confidence)

    # =====================================================
    # LANGUAGE
    # =====================================================
    if extracted_persona.language:
        lang = extracted_persona.language

        add_signal(category="preference", field="language", value=lang.language, confidence=lang.confidence)
        add_signal(category="preference", field="complexity", value=lang.complexity, confidence=lang.confidence)
        add_signal(category="preference", field="jargon_policy", value=lang.jargon_policy, confidence=lang.confidence)

    # =====================================================
    # CONSTRAINTS
    # =====================================================
    if extracted_persona.constraints:
        cons = extracted_persona.constraints

        add_signal(category="constraint", field="constraints", value=cons.constraints, confidence=cons.confidence)

    return signals


# =========================================================
# PROJECTION BASED ON COGNITION DECISIONS (FIELD-ACCURATE)
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


def project_persona_by_decisions(
    extracted_persona: UserPersonaModel,
    decisions: list[dict],
) -> UserPersonaModel | None:
    """
    Project a partial persona based on cognition decisions.

    - Field-level accurate
    - Block-safe
    - No silent carryover
    """

    projected_blocks: dict = {}
    any_commit = False

    for decision in decisions:
        if decision.get("action") not in {"COMMIT", "PARTIAL_COMMIT"}:
            continue
        if decision.get("target") != "persona":
            continue

        for field in decision.get("scope", []):
            mapping = FIELD_TO_BLOCK.get(field)
            if not mapping:
                continue

            block_name, _ = mapping
            source_block = getattr(extracted_persona, block_name, None)
            if not source_block:
                continue

            if block_name not in projected_blocks:
                block = source_block.model_copy(deep=True)

                # remove learning-only metadata
                if hasattr(block, "confidence"):
                    block.confidence = None

                projected_blocks[block_name] = block


            any_commit = True

    if not any_commit:
        return None

    return UserPersonaModel(**projected_blocks)
