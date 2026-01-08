# persona/persona_adapters.py

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel


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

        add_signal(
            category="identity",
            field="job_title",
            value=ui.job_title,
            confidence=ui.confidence,
        )

        add_signal(
            category="identity",
            field="seniority",
            value=ui.seniority,
            confidence=ui.confidence,
        )

        add_signal(
            category="identity",
            field="function",
            value=ui.function,
            confidence=ui.confidence,
        )

        add_signal(
            category="identity",
            field="decision_authority",
            value=ui.decision_authority,
            confidence=ui.confidence,
        )

    # =====================================================
    # COMPANY — PROFILE
    # =====================================================
    if extracted_persona.company_profile:
        cp = extracted_persona.company_profile

        add_signal(
            category="organization",
            field="company_name",
            value=cp.company_name,
            confidence=cp.confidence,
        )

        add_signal(
            category="organization",
            field="industry",
            value=cp.industry,
            confidence=cp.confidence,
        )

        add_signal(
            category="organization",
            field="company_size",
            value=cp.company_size,
            confidence=cp.confidence,
        )

        add_signal(
            category="organization",
            field="company_stage",
            value=cp.company_stage,
            confidence=cp.confidence,
        )

    # =====================================================
    # COMPANY — BUSINESS
    # =====================================================
    if extracted_persona.company_business:
        cb = extracted_persona.company_business

        add_signal(
            category="organization",
            field="business_model",
            value=cb.business_model,
            confidence=cb.confidence,
        )

        add_signal(
            category="organization",
            field="sales_motion",
            value=cb.sales_motion,
            confidence=cb.confidence,
        )

        add_signal(
            category="organization",
            field="target_customers",
            value=cb.target_customers,
            confidence=cb.confidence,
        )

    # =====================================================
    # COMPANY — PRODUCTS / TECH
    # =====================================================
    if extracted_persona.company_products:
        cprod = extracted_persona.company_products

        add_signal(
            category="organization",
            field="products",
            value=cprod.products,
            confidence=cprod.confidence,
        )

        add_signal(
            category="organization",
            field="tech_orientation",
            value=cprod.tech_orientation,
            confidence=cprod.confidence,
        )

    # =====================================================
    # OBJECTIVE
    # =====================================================
    if extracted_persona.objective:
        obj = extracted_persona.objective

        add_signal(
            category="preference",
            field="primary_goal",
            value=obj.primary_goal,
            confidence=obj.confidence,
        )

    # =====================================================
    # CONTENT FORMAT
    # =====================================================
    if extracted_persona.content_format:
        cf = extracted_persona.content_format

        add_signal(
            category="preference",
            field="length_preference",
            value=cf.length_preference,
            confidence=cf.confidence,
        )

        add_signal(
            category="preference",
            field="preferred_format",
            value=cf.preferred_format,
            confidence=cf.confidence,
        )

    # =====================================================
    # AUDIENCE
    # =====================================================
    if extracted_persona.audience:
        aud = extracted_persona.audience

        add_signal(
            category="preference",
            field="audience_type",
            value=aud.audience_type,
            confidence=aud.confidence,
        )

        add_signal(
            category="preference",
            field="audience_level",
            value=aud.audience_level,
            confidence=aud.confidence,
        )

    # =====================================================
    # TONE
    # =====================================================
    if extracted_persona.tone:
        tone = extracted_persona.tone

        add_signal(
            category="preference",
            field="tone",
            value=tone.tone,
            confidence=tone.confidence,
        )

        add_signal(
            category="preference",
            field="voice",
            value=tone.voice,
            confidence=tone.confidence,
        )

    # =====================================================
    # WRITING STYLE
    # =====================================================
    if extracted_persona.writing_style:
        ws = extracted_persona.writing_style

        add_signal(
            category="preference",
            field="style",
            value=ws.style,
            confidence=ws.confidence,
        )

    # =====================================================
    # LANGUAGE
    # =====================================================
    if extracted_persona.language:
        lang = extracted_persona.language

        add_signal(
            category="preference",
            field="complexity",
            value=lang.complexity,
            confidence=lang.confidence,
        )

    # =====================================================
    # CONSTRAINTS
    # =====================================================
    if extracted_persona.constraints:
        cons = extracted_persona.constraints

        add_signal(
            category="constraint",
            field="constraints",
            value=cons.constraints,
            confidence=cons.confidence,
        )

    return signals


# =========================================================
# PROJECTION BASED ON COGNITION DECISIONS
# =========================================================

def project_persona_by_decisions(
    extracted_persona: UserPersonaModel,
    decisions: list[dict],
) -> UserPersonaModel | None:
    """
    Project a partial persona based on cognition decisions.

    - Only fields explicitly approved by cognition are projected
    - No silent carryover
    """

    allowed_fields: set[str] = set()

    for decision in decisions:
        if (
            decision.get("action") in {"COMMIT", "PARTIAL_COMMIT"}
            and decision.get("target") == "persona"
        ):
            allowed_fields.update(decision.get("scope", []))

    if not allowed_fields:
        return None

    return UserPersonaModel(
        # Identity
        user_identity=extracted_persona.user_identity if "user_identity" in allowed_fields else None,

        # Company
        company_profile=extracted_persona.company_profile if "company_profile" in allowed_fields else None,
        company_business=extracted_persona.company_business if "company_business" in allowed_fields else None,
        company_products=extracted_persona.company_products if "company_products" in allowed_fields else None,
        company_brand=extracted_persona.company_brand if "company_brand" in allowed_fields else None,

        # Behavioral
        objective=extracted_persona.objective if "objective" in allowed_fields else None,
        content_format=extracted_persona.content_format if "content_format" in allowed_fields else None,
        audience=extracted_persona.audience if "audience" in allowed_fields else None,
        tone=extracted_persona.tone if "tone" in allowed_fields else None,
        writing_style=extracted_persona.writing_style if "writing_style" in allowed_fields else None,
        language=extracted_persona.language if "language" in allowed_fields else None,
        constraints=extracted_persona.constraints if "constraints" in allowed_fields else None,
    )
