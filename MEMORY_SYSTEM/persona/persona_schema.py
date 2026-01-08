from typing import Optional, List, Literal
from pydantic import BaseModel, Field

# =========================================================
# 1. USER IDENTITY (WHO THE USER IS)
# =========================================================

class UserIdentityPersona(BaseModel):
    full_name: Optional[str] = None
    preferred_name: Optional[str] = None

    job_title: Optional[str] = None
    seniority: Optional[Literal[
        "founder",
        "c_level",
        "vp",
        "director",
        "manager",
        "individual_contributor"
    ]] = None

    function: Optional[str] = Field(
        None, description="Marketing, Engineering, Sales, Product, Ops, etc."
    )

    decision_authority: Optional[Literal[
        "decision_maker",
        "influencer",
        "executor"
    ]] = None

    years_experience: Optional[int] = None

    confidence: float = Field(ge=0.0, le=1.0)


# =========================================================
# 2. COMPANY / ORGANIZATION CONTEXT
# =========================================================

class CompanyProfilePersona(BaseModel):
    company_name: Optional[str] = None
    registered_name: Optional[str] = None

    description: Optional[str] = None
    industry: Optional[str] = None

    company_stage: Optional[Literal[
        "startup",
        "growth",
        "enterprise"
    ]] = None

    company_size: Optional[str] = None  # e.g. "1-10", "11-50"
    headquarters: Optional[str] = None
    geo_market: Optional[str] = None

    website: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)


class CompanyBusinessPersona(BaseModel):
    business_model: Optional[str] = Field(
        None, description="B2B SaaS, B2C, Marketplace, Services, Hybrid"
    )

    pricing_model: Optional[str] = None
    sales_motion: Optional[Literal[
        "self_serve",
        "sales_led",
        "hybrid"
    ]] = None

    target_customers: Optional[List[str]] = None
    target_industries: Optional[List[str]] = None

    core_value_proposition: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)


class CompanyProductPersona(BaseModel):
    products: Optional[List[dict]] = Field(
        None,
        description="Each product: name, category, positioning"
    )

    tech_orientation: Optional[Literal[
        "ai_first",
        "tech_enabled",
        "non_technical"
    ]] = None

    differentiation_axes: Optional[List[str]] = None

    confidence: float = Field(ge=0.0, le=1.0)


class CompanyBrandPersona(BaseModel):
    core_values: Optional[List[str]] = None

    brand_personality: Optional[str] = Field(
        None, description="Innovative, Trustworthy, Bold, Premium, Friendly"
    )

    compliance_sensitivity: Optional[bool] = None
    data_security_sensitivity: Optional[bool] = None

    confidence: float = Field(ge=0.0, le=1.0)


# =========================================================
# 3. BEHAVIORAL / CONTENT PERSONAS (YOUR ORIGINAL MODEL)
# =========================================================

class ObjectivePersona(BaseModel):
    primary_goal: Optional[str] = Field(
        None, description="inform, persuade, sell, educate, document"
    )
    desired_action: Optional[str] = None
    success_criteria: Optional[str] = None
    horizon: Optional[Literal["short_term", "long_term"]] = None

    confidence: float = Field(ge=0.0, le=1.0)


class ContentFormatPersona(BaseModel):
    content_types: Optional[List[str]] = None
    preferred_format: Optional[str] = None
    length_preference: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)


class AudiencePersona(BaseModel):
    audience_type: Optional[str] = None
    audience_domain: Optional[str] = None
    audience_level: Optional[Literal[
        "beginner",
        "intermediate",
        "expert"
    ]] = None
    geo_context: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)


class TonePersona(BaseModel):
    tone: Optional[str] = None
    voice: Optional[Literal[
        "first_person",
        "second_person",
        "third_person"
    ]] = None
    emotional_intensity: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)


class WritingStylePersona(BaseModel):
    style: Optional[str] = None
    sentence_structure: Optional[str] = None
    use_examples: Optional[bool] = None
    use_storytelling: Optional[bool] = None

    confidence: float = Field(ge=0.0, le=1.0)


class LanguagePersona(BaseModel):
    language: Optional[str] = None
    complexity: Optional[str] = None
    jargon_policy: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)


class ConstraintPersona(BaseModel):
    constraints: Optional[List[str]] = None

    confidence: float = Field(ge=0.0, le=1.0)


# =========================================================
# 4. AGGREGATED PERSONA MODEL (FINAL)
# =========================================================

class UserPersonaModel(BaseModel):
    """
    Full user persona model.

    Design principles:
    - Incremental learning
    - Partial updates only
    - Confidence-driven merging
    - Identity & company separated from behavior
    """

    # Identity
    user_identity: Optional[UserIdentityPersona] = None

    # Company context
    company_profile: Optional[CompanyProfilePersona] = None
    company_business: Optional[CompanyBusinessPersona] = None
    company_products: Optional[CompanyProductPersona] = None
    company_brand: Optional[CompanyBrandPersona] = None

    # Behavioral / content personas
    objective: Optional[ObjectivePersona] = None
    content_format: Optional[ContentFormatPersona] = None
    audience: Optional[AudiencePersona] = None
    tone: Optional[TonePersona] = None
    writing_style: Optional[WritingStylePersona] = None
    language: Optional[LanguagePersona] = None
    constraints: Optional[ConstraintPersona] = None
