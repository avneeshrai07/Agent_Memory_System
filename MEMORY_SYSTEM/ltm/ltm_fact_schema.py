from typing import Literal, List
from pydantic import BaseModel, Field


# -----------------------------
# LTM confidence metadata (LLM-owned)
# -----------------------------
class LTMConfidence(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident the system is that this memory is correct."
    )

    source: Literal["explicit", "implicit", "derived", "validated"] = Field(
        ...,
        description="How this memory was learned."
    )


# -----------------------------
# LTM memory (EXTRACTION ONLY)
# -----------------------------
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
        description="Semantic category of the extracted memory."
    )

    topic: str = Field(
        ...,
        description="Short, reusable label for the memory."
    )

    fact: str = Field(
        ...,
        description="Atomic factual statement. Must stand alone without context."
    )

    importance: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="How important this memory is for future reasoning."
    )

    confidence: LTMConfidence = Field(
        ...,
        description="Confidence metadata for this memory."
    )


class LTMMemoryExtractionBatch(BaseModel):
    facts: List[LTMMemoryExtraction] = Field(
        ...,
        description="List of extracted long-term memory facts."
    )