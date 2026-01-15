# MEMORY_SYSTEM/artifacts/artifact_store.py

import uuid
from datetime import datetime, timezone

class ArtifactStore:
    def __init__(self, *, repo, s3_client):
        self.repo = repo
        self.s3 = s3_client

    async def create_artifact(
        self,
        *,
        user_id: str,
        route: str,
        artifact_type: str,
        content: str,
        summary: str,
    ) -> dict:
        """
        Atomic artifact creation:
        LLM → S3 → Postgres
        """
        artifact_id = str(uuid.uuid4())

        # 1️⃣ Write content to S3
        content_ref = await self.s3.write_content(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            content=content,
        )

        # 2️⃣ Persist metadata to Postgres
        artifact = await self.repo.create_artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            summary=summary,
            content_ref=content_ref,
            metadata={
                "source": "llm",
                "created_by": user_id,
                "route": route,
            },
            created_at=datetime.now(timezone.utc),
        )

        return artifact
