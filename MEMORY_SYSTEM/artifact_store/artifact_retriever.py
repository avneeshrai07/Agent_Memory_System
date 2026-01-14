# artifacts/artifact_retriever.py
from MEMORY_SYSTEM.artifact_store.artifact_models import Artifact

class ArtifactRetriever:
    def __init__(self, store):
        # store injected (DB, S3, etc.)
        self._store = store

    def get_latest(self, artifact_id: str) -> Artifact:
        versions = self._store.get(artifact_id)
        if not versions:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return versions[-1]

    def get_version(self, artifact_id: str, version: int) -> Artifact:
        versions = self._store.get(artifact_id)
        if not versions or version > len(versions):
            raise KeyError("Artifact version not found")
        return versions[version - 1]
