import asyncio
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from MEMORY_SYSTEM.artifacts.s3_client import ArtifactS3Client


async def test_s3_upload():
    print("\n[S3 TEST] Starting S3 artifact upload test")

    s3_client = ArtifactS3Client(
        bucket="memory-artifact-storage"
    )

    artifact_id = str(uuid.uuid4())
    artifact_type = "email"

    content = f"""
# Test Email Artifact

This is a test artifact uploaded at {datetime.now(timezone.utc).isoformat()}.

Artifact ID: {artifact_id}
"""

    content_ref = await s3_client.write_content(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        content=content,
    )

    print("[S3 TEST] Upload successful")
    print("[S3 TEST] content_ref:", content_ref)

    assert content_ref.startswith(
        "s3://memory-artifact-storage-old/artifacts/"
    )
    assert artifact_id in content_ref
    assert content_ref.endswith(".md")

    print("[S3 TEST] content_ref format validated")


if __name__ == "__main__":
    asyncio.run(test_s3_upload())
