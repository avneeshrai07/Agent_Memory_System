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

    embedding_dim: int = 1024
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    extraction_model_id: str = "amazon.nova-lite-v1:0"

    @classmethod
    def from_env(cls) -> MemoryConfig:
        return cls(
            postgres_dsn=os.environ["POSTGRES_DSN"],
            redis_url=os.environ["REDIS_URL"],
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            embedding_dim=int(os.environ.get("AGENT_MEMORY_EMBEDDING_DIM", "1024")),
        )
