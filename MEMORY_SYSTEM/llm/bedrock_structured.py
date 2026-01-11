from dotenv import load_dotenv
load_dotenv()

from typing import Optional, Type
import traceback
import os

from langchain_aws import ChatBedrock
from pydantic import BaseModel


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_MODEL_REGION = "ap-southeast-2"
LLM_MODEL_NEWS_FETCHER = "amazon.nova-lite-v1:0"

# -------------------------------------------------------------------
# BEDROCK INITIALIZATION
# -------------------------------------------------------------------

try:
    llm = ChatBedrock(
        model_id=LLM_MODEL_NEWS_FETCHER,
        region_name=AWS_MODEL_REGION,
        temperature=0.1,
        max_tokens=9999,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
except Exception:
    print("❌ [BEDROCK] Failed to initialize ChatBedrock")
    traceback.print_exc()
    llm = None


# -------------------------------------------------------------------
# STRUCTURED LLM CALL WITH SAFETY
# -------------------------------------------------------------------

async def bedrock_structured_llm_call(
    system_prompt: str,
    user_prompt: str,
    output_structure: Type[BaseModel],
    model_dump: bool = False,
) -> Optional[dict]:
    """
    Safe structured LLM call wrapper.

    Guarantees:
    - Always returns dict or None
    - Never raises uncaught exceptions
    - Full traceback on failure
    """

    if llm is None:
        print("❌ [BEDROCK] LLM not initialized")
        return None

    try:
        structured_llm = llm.with_structured_output(output_structure)
    except Exception:
        print("❌ [BEDROCK] Failed to attach structured output")
        traceback.print_exc()
        return None

    try:
        response = await structured_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception:
        print("❌ [BEDROCK] LLM invocation failed")
        traceback.print_exc()
        return None

    if response is None:
        print("⚠️ [BEDROCK] Null response from model")
        return None

    try:
        if model_dump:
            return response.model_dump()
        return response
    except Exception:
        print("❌ [BEDROCK] Failed to serialize model output")
        traceback.print_exc()
        return None