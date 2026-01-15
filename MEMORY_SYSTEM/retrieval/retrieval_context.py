# MEMORY_SYSTEM/retrieval/retrieval_context.py

from typing import TypedDict, List, Optional
from MEMORY_SYSTEM.direct_context.dc_types import DerivedCurrentContext
from MEMORY_SYSTEM.artifacts.artifact_contract import Artifact


class RetrievalContext(TypedDict):
    # Exactly ONE of these will be populated per turn
    dcc: Optional[DerivedCurrentContext]
    artifact: Optional[Artifact]
    artifact_summaries: Optional[List[dict]]
