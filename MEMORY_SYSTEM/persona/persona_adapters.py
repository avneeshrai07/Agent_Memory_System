from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel


from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel


def persona_to_signals(extracted_persona: UserPersonaModel) -> list[dict]:
    """
    Convert structured persona blocks into cognition signal candidates.
    This function DISTILLS rich blocks into scalar cognition signals.
    """

    signals: list[dict] = []

    def add_signal(category: str, field: str, value, confidence: float):
        if value is None:
            return
        signals.append({
            "category": category,
            "field": field,
            "value": value,
            "base_confidence": confidence,
            "source": "derived",
            "frequency": 1,
        })

    # -----------------------------
    # OBJECTIVE
    # -----------------------------
    if extracted_persona.objective:
        add_signal(
            "preference",
            "objective",
            extracted_persona.objective.primary_goal,
            extracted_persona.objective.confidence,
        )

    # -----------------------------
    # CONTENT FORMAT
    # -----------------------------
    if extracted_persona.content_format:
        add_signal(
            "preference",
            "content_format",
            extracted_persona.content_format.length_preference,
            extracted_persona.content_format.confidence,
        )

    # -----------------------------
    # AUDIENCE
    # -----------------------------
    if extracted_persona.audience:
        add_signal(
            "preference",
            "audience",
            extracted_persona.audience.audience_type,
            extracted_persona.audience.confidence,
        )

    # -----------------------------
    # TONE
    # -----------------------------
    if extracted_persona.tone:
        add_signal(
            "preference",
            "tone",
            extracted_persona.tone.tone,
            extracted_persona.tone.confidence,
        )

    # -----------------------------
    # WRITING STYLE
    # -----------------------------
    if extracted_persona.writing_style:
        add_signal(
            "preference",
            "writing_style",
            extracted_persona.writing_style.style,
            extracted_persona.writing_style.confidence,
        )

    # -----------------------------
    # LANGUAGE
    # -----------------------------
    if extracted_persona.language:
        add_signal(
            "preference",
            "language",
            extracted_persona.language.complexity,
            extracted_persona.language.confidence,
        )

    # -----------------------------
    # CONSTRAINTS (OPTIONAL)
    # -----------------------------
    if extracted_persona.constraints:
        add_signal(
            "constraint",
            "constraints",
            "present",
            extracted_persona.constraints.confidence,
        )

    return signals


def project_persona_by_decisions(
    extracted_persona: UserPersonaModel,
    decisions: list[dict],
) -> UserPersonaModel | None:
    """
    Project a partial persona based on cognition decisions.
    Adapter layer: Cognition → Persona.
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
        objective=extracted_persona.objective if "objective" in allowed_fields else None,
        content_format=extracted_persona.content_format if "content_format" in allowed_fields else None,
        audience=extracted_persona.audience if "audience" in allowed_fields else None,
        tone=extracted_persona.tone if "tone" in allowed_fields else None,
        writing_style=extracted_persona.writing_style if "writing_style" in allowed_fields else None,
        language=extracted_persona.language if "language" in allowed_fields else None,
        constraints=extracted_persona.constraints if "constraints" in allowed_fields else None,
    )
