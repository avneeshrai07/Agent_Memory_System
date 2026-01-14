# artifacts/artifact_inserter.py
from datetime import datetime
from typing import Dict, List
from MEMORY_SYSTEM.artifact_store.artifact_models import Artifact

class ArtifactInserter:
    def __init__(self):
        # replace with DB / S3 / Postgres later
        self._store: Dict[str, List[Artifact]] = {}

    def insert_new(
        self,
        artifact_id: str,
        artifact_type: str,
        content: str,
        change_reason: str | None = None
    ) -> Artifact:
        versions = self._store.get(artifact_id, [])
        new_version = len(versions) + 1

        artifact = Artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            version=new_version,
            content=content,
            change_reason=change_reason,
            created_at=datetime.utcnow()
        )

        versions.append(artifact)
        self._store[artifact_id] = versions

        return artifact
