from typing import List, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field


ConfidenceScore = Annotated[
    float,
    Field(ge=0.0, le=1.0)
]


class LTMFact(BaseModel):
    fact: str
    memory_type: str
    confidence_score: ConfidenceScore
    semantic_topic: Optional[str] = None


class LTMFactList(BaseModel):
    """
    Bedrock-compatible tool input schema
    """
    facts: List[LTMFact] = Field(
        default_factory=list,
        description="List of extracted long-term memory facts"
    )
