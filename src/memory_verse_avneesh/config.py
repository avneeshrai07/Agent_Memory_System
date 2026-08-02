"""Settings objects for the library. No global/module-level state — a host
application constructs and passes these explicitly."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    """Config a host application builds once and uses to wire up the
    concrete Postgres/Redis(-or-Upstash)/Bedrock backends. The library
    itself never reads environment variables directly — only this class's
    from_env() does, and only because it's a convenience, not a requirement.
    """

    postgres_dsn: str

    # Tier 0/1 cache backend — exactly one of these two must be set:
    # standard Redis protocol (redis_url) or Upstash's REST API
    # (upstash_url + upstash_token). Validated in __post_init__ so a
    # misconfiguration fails at startup with a clear message, not three
    # calls deep into a None where a client was expected.
    redis_url: str | None = None
    upstash_url: str | None = None
    upstash_token: str | None = None

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

    def __post_init__(self) -> None:
        has_redis = self.redis_url is not None
        has_upstash_url = self.upstash_url is not None
        has_upstash_token = self.upstash_token is not None

        if has_upstash_url != has_upstash_token:
            raise ValueError(
                "MemoryConfig: upstash_url and upstash_token must both be set, or "
                "neither — got only one."
            )

        has_upstash = has_upstash_url and has_upstash_token

        if has_redis and has_upstash:
            raise ValueError(
                "MemoryConfig: both redis_url and upstash_url/upstash_token are set — "
                "pick one Tier 0/1 cache backend, not both."
            )
        if not has_redis and not has_upstash:
            raise ValueError(
                "MemoryConfig: no Tier 0/1 cache backend configured — set redis_url "
                "(standard Redis) or both upstash_url and upstash_token (Upstash)."
            )

    @classmethod
    def from_env(cls) -> MemoryConfig:
        return cls(
            postgres_dsn=os.environ["POSTGRES_DSN"],
            redis_url=os.environ.get("REDIS_URL"),
            upstash_url=os.environ.get("UPSTASH_REDIS_REST_URL"),
            upstash_token=os.environ.get("UPSTASH_REDIS_REST_TOKEN"),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_LLM_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_LLM_SECRET_ACCESS_KEY"),
            embedding_dim=int(os.environ.get("MEMORY_VERSE_AVNEESH_EMBEDDING_DIM", "1024")),
        )
