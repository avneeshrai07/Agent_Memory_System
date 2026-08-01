"""Settings objects for the library. No global/module-level state — a host
application constructs and passes these explicitly."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    """Config a host application builds once and uses to wire up the
    concrete Postgres/Redis/Bedrock backends. The library itself never reads
    environment variables directly — only this class's from_env() does, and
    only because it's a convenience, not a requirement.
    """

    postgres_dsn: str
    redis_url: str
    aws_region: str = "us-east-1"

    # Optional: only needed when Bedrock should use credentials scoped
    # separately from whatever else in the environment reads the standard
    # AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY names (which boto3's default
    # chain already discovers on its own). Leave unset to fall back to that
    # default chain — the common production pattern (instance/task role).
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    embedding_dim: int = 1024
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    extraction_model_id: str = "amazon.nova-lite-v1:0"

    @classmethod
    def from_env(cls) -> MemoryConfig:
        return cls(
            postgres_dsn=os.environ["POSTGRES_DSN"],
            redis_url=os.environ["REDIS_URL"],
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_LLM_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_LLM_SECRET_ACCESS_KEY"),
            embedding_dim=int(os.environ.get("AGENT_MEMORY_EMBEDDING_DIM", "1024")),
        )
