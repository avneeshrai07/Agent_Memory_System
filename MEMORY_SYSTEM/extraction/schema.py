# MEMORY_SYSTEM/extraction/schema.py

from pydantic import BaseModel
from typing import Literal

class ExtractedFact(BaseModel):
    topic: str
    fact: str
    category: Literal[
        "technical_context",
        "preference",
        "constraint",
        "domain",
        "expertise"
    ]
    importance: int
    signal_type: Literal["explicit", "implicit", "derived"]
