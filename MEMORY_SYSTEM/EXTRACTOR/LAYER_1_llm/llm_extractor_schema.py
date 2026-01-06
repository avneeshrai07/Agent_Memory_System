from typing import List, Literal, Annotated
from pydantic import BaseModel, Field


Importance = Annotated[int, Field(ge=1, le=10)]


class ExtractedFact(BaseModel):
    topic: str = Field(
        ...,
        description="Main subject area of the fact (e.g., database, API design, AI tooling)"
    )

    fact: str = Field(
        ...,
        description="Clear, concise statement of the extracted fact"
    )

    category: Literal[
        "technical_context",
        "user_preference",
        "problem_domain",
        "expertise",
        "constraint",
        "learned_pattern"
    ] = Field(
        ...,
        description="Type of fact extracted"
    )

    importance: Importance = Field(
        ...,
        description="Importance of the fact on a scale of 1 (low) to 10 (critical)"
    )


class FactExtractionOutput(BaseModel):
    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="List of extracted facts from the conversation"
    )
