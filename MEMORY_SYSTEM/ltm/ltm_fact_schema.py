from typing import Literal, List
from pydantic import BaseModel, Field


# =====================================================
# FACTUAL LTM CONFIDENCE (LLM-OWNED)
# =====================================================
class LTMConfidence(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident the system is that this factual memory is correct."
    )

    source: Literal["explicit", "implicit", "derived", "validated"] = Field(
        ...,
        description="How this factual memory was learned."
    )


# =====================================================
# EPISODIC CONFIDENCE (LLM-OWNED, LIGHTER WEIGHT)
# =====================================================
class EpisodicConfidence(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that this episodic context is correct. "
            "Lower bar than factual truth; reflects conversational certainty."
        )
    )

    source: Literal["explicit", "implicit", "inferred"] = Field(
        ...,
        description="How this episodic context was identified."
    )


# =====================================================
# FACTUAL LTM EXTRACTION (EXTRACTION ONLY)
# =====================================================
class LTMMemoryExtraction(BaseModel):
    category: Literal[
        "technical_context",
        "problem_domain",
        "constraint",
        "preference",
        "expertise",
        "validated_outcome",
        "learned_pattern"
    ] = Field(
        ...,
        description="Semantic category of the extracted factual long-term memory."
    )

    topic: str = Field(
        ...,
        description="Short, reusable label for the factual memory."
    )

    fact: str = Field(
        ...,
        description="Atomic factual statement. Must stand alone without context."
    )

    importance: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="How important this factual memory is for future reasoning."
    )

    confidence: LTMConfidence = Field(
        ...,
        description="Confidence metadata for this factual memory."
    )


# =====================================================
# EPISODIC LTM EXTRACTION (EXTRACTION ONLY)
# =====================================================
class LTMEpisodicExtraction(BaseModel):
    context_type: Literal[
        "entity_binding",
        "referential_alias",
        "ongoing_goal",
        "active_artifact",
        "conversation_focus",
        "role_assignment"
    ] = Field(
        ...,
        description=(
            "Type of episodic context being captured. "
            "Used to resolve references and maintain continuity across turns."
        )
    )

    key: str = Field(
        ...,
        description=(
            "Identifier for the episodic context. "
            "Examples: 'famous_personality', 'current_goal', 'email_target'."
        )
    )

    value: str = Field(
        ...,
        description=(
            "Resolved meaning of the episodic context. "
            "Must be explicit and unambiguous."
        )
    )

    scope: Literal[
        "session",
        "multi_turn",
        "task"
    ] = Field(
        ...,
        description=(
            "Expected lifespan of this episodic context. "
            "Session = current conversation only, "
            "Multi_turn = short continuation, "
            "Task = until task completion."
        )
    )

    confidence: EpisodicConfidence = Field(
        ...,
        description="Confidence metadata for this episodic context."
    )


# =====================================================
# UNIFIED LTM EXTRACTION BATCH (LLM OUTPUT)
# =====================================================
class LTMMemoryExtractionBatch(BaseModel):
    facts: List[LTMMemoryExtraction] = Field(
        default_factory=list,
        description="Extracted factual long-term memories."
    )

    episodic: List[LTMEpisodicExtraction] = Field(
        default_factory=list,
        description="Extracted episodic (referential / narrative) memories."
    )
