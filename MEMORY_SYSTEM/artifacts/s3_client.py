import boto3
import os
import asyncio
from dotenv import load_dotenv

# Load AWS credentials once
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")


class ArtifactS3Client:
    def __init__(self, bucket: str):
        self.bucket = bucket

        # Explicit credentials (same pattern as your reference)
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION,
        )

    def _put_object_sync(
        self,
        *,
        key: str,
        content: str,
    ) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/markdown",
        )

    async def write_content(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        content: str,
    ) -> str:
        """
        Async-safe wrapper using boto3 (sync) + thread offloading.
        """
        key = f"artifacts/{artifact_type}/{artifact_id}.md"

        await asyncio.to_thread(
            self._put_object_sync,
            key=key,
            content=content,
        )

        return f"s3://{self.bucket}/{key}"
