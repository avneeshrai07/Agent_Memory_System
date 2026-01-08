from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class ObjectivePersona(BaseModel):
    primary_goal: Optional[str] = Field(
        None, description="inform, persuade, sell, educate, document"
    )
    desired_action: Optional[str] = Field(
        None, description="Desired user action (CTA)"
    )
    success_criteria: Optional[str] = Field(
        None, description="How success is measured"
    )
    horizon: Optional[Literal["short_term", "long_term"]] = None

    confidence: float = Field(ge=0.0, le=1.0)


class ContentFormatPersona(BaseModel):
    content_types: Optional[List[str]] = Field(
        None, description="email, blog, proposal, SOP, etc."
    )
    preferred_format: Optional[str] = Field(
        None, description="paragraphs, bullets, steps, table"
    )
    length_preference: Optional[str] = Field(
        None, description="short, medium, long"
    )

    confidence: float = Field(ge=0.0, le=1.0)


class AudiencePersona(BaseModel):
    audience_type: Optional[str] = Field(
        None, description="internal, client, executive, public"
    )
    audience_domain: Optional[str] = Field(
        None, description="industry or domain"
    )
    audience_level: Optional[Literal["beginner", "intermediate", "expert"]] = None
    geo_context: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)


class TonePersona(BaseModel):
    tone: Optional[str] = Field(
        None, description="formal, professional, casual, persuasive"
    )
    voice: Optional[Literal["first_person", "second_person", "third_person"]] = None
    emotional_intensity: Optional[str] = Field(
        None, description="neutral, confident, enthusiastic, assertive"
    )

    confidence: float = Field(ge=0.0, le=1.0)


class WritingStylePersona(BaseModel):
    style: Optional[str] = Field(
        None, description="plain, analytical, narrative, instructional"
    )
    sentence_structure: Optional[str] = Field(
        None, description="short, long, mixed"
    )
    use_examples: Optional[bool] = None
    use_storytelling: Optional[bool] = None

    confidence: float = Field(ge=0.0, le=1.0)


class LanguagePersona(BaseModel):
    language: Optional[str] = None
    complexity: Optional[str] = Field(
        None, description="simple, professional, academic"
    )
    jargon_policy: Optional[str] = Field(
        None, description="avoid, allowed, required"
    )

    confidence: float = Field(ge=0.0, le=1.0)


class ConstraintPersona(BaseModel):
    constraints: Optional[List[str]] = Field(
        None, description="Things that must be avoided or followed"
    )

    confidence: float = Field(ge=0.0, le=1.0)


# -------------------------------------------------------------------
# ✅ AGGREGATED USER PERSONA MODEL (CORRECT)
# -------------------------------------------------------------------

class UserPersonaModel(BaseModel):
    """
    Aggregate persona model.

    Each field is optional because:
    - Persona is learned incrementally
    - Absence means 'unknown', not null preference
    """

    objective: Optional[ObjectivePersona] = None
    content_format: Optional[ContentFormatPersona] = None
    audience: Optional[AudiencePersona] = None
    tone: Optional[TonePersona] = None
    writing_style: Optional[WritingStylePersona] = None
    language: Optional[LanguagePersona] = None
    constraints: Optional[ConstraintPersona] = None
