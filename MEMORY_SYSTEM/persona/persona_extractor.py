"""
Persona Extractor (Structured Output)
====================================

Purpose:
- Extract user persona signals from a single interaction
- Use Bedrock structured output with Pydantic
- NO manual JSON parsing
- NO schema prompting
- NO normalization hacks

This file is intentionally simple.
"""

from typing import Optional
import traceback

from langchain_aws import ChatBedrock

from MEMORY_SYSTEM.persona.persona_schema import UserPersonaModel


# -------------------------------------------------------------------
# SYSTEM PROMPT (PERSONA INFERENCE ONLY)
# -------------------------------------------------------------------

PERSONA_EXTRACTION_SYSTEM_PROMPT = """
You are an AI system that infers persistent user preferences and writing expectations.

Your task:
- Infer preferences ONLY if they are clearly implied
- Do NOT guess or over-infer
- Leave fields null if uncertain
- Confidence should reflect how strongly the signal is implied

Rules:
- Extract preferences, not the task itself
- Do not restate the user's request
- Do not include explanations
"""


# -------------------------------------------------------------------
# PERSONA EXTRACTOR
# -------------------------------------------------------------------

class PersonaExtractor:
    """
    Persona extractor using Bedrock structured output.
    """

    def __init__(self, llm: ChatBedrock):
        self.llm = llm

    async def extract(
        self,
        user_prompt: str
    ) -> Optional[UserPersonaModel]:
        """
        Extract persona from a single user interaction.

        Returns:
            UserPersonaModel or None
        """

        print("[persona_extractor] starting extraction")

        try:
            structured_llm = self.llm.with_structured_output(UserPersonaModel)

            persona = await structured_llm.ainvoke(
                [
                    {"role": "system", "content": PERSONA_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )

            if persona is None:
                print("[persona_extractor] no persona returned")
                return None

            # Defensive: check if at least one field is populated
            if not any(
                getattr(persona, field) is not None
                for field in persona.model_fields
            ):
                print("[persona_extractor] persona empty, skipping")
                return None

            print("[persona_extractor] persona extracted successfully")
            print(persona.model_dump())

            return persona

        except Exception:
            print("[persona_extractor] extraction failed")
            print(traceback.format_exc())
            return None
