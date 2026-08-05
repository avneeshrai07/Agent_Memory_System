"""Settings objects for the library. No global/module-level state — a host
application constructs and passes these explicitly."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MemoryConfig:
    """Config a host application builds once and uses to wire up the
    concrete Postgres/Redis(-or-Upstash)/Bedrock backends. The library
    itself never reads environment variables directly — only this class's
    from_env() does, and only because it's a convenience, not a requirement.

    Most hosts won't build this directly — memory_verse_avneesh.connect()
    takes the same fields and returns a fully wired Memory object (pool
    created, every table's ensure_schema() called, every client
    constructed) in one call. Build MemoryConfig by hand only for custom
    wiring connect() doesn't cover.
    """

    # Postgres: exactly one of these two shapes must be set — a single
    # database_url, or the granular host/port/user/password/database
    # fields (mirroring storage-verse-avneesh's own dsn-vs-host duality,
    # for hosts that don't have a single connection string to hand).
    database_url: str | None = None
    postgres_host: str | None = None
    postgres_port: int | None = None
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_database: str | None = None

    # No default, deliberately: see PostgresFactStore's own docstring on why
    # a silent "public" default is a real cross-tenant-collision risk, not
    # just a style preference. Safe to require here despite coming after
    # fields with defaults above -- kw_only=True on the whole dataclass
    # (Python 3.11+) lifts the usual "no required field after a defaulted
    # one" ordering rule, so this still raises a clear TypeError if omitted.
    postgres_schema: str

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
    # Named aws_llm_* (not the plain aws_* boto3 uses) to match the AWS_LLM_*
    # env vars from_env() already read — credentials scoped to this
    # library's own LLM calls, not whatever else in the process might read
    # the standard AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY names.
    aws_llm_access_key_id: str | None = None
    aws_llm_secret_access_key: str | None = None

    embedding_dim: int = 1024
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    extraction_model_id: str = "amazon.nova-lite-v1:0"

    def __post_init__(self) -> None:
        has_database_url = self.database_url is not None
        has_full_host_config = bool(
            self.postgres_host and self.postgres_user
            and self.postgres_password and self.postgres_database
        )
        has_partial_host_config = (
            any([self.postgres_host, self.postgres_user, self.postgres_password, self.postgres_database])
            and not has_full_host_config
        )

        if has_partial_host_config:
            raise ValueError(
                "MemoryConfig: postgres_host/postgres_user/postgres_password/"
                "postgres_database must all be set together, or none of them — "
                "got only some."
            )
        if has_database_url and has_full_host_config:
            raise ValueError(
                "MemoryConfig: both database_url and the postgres_host/user/"
                "password/database fields are set — pick one way to configure "
                "Postgres, not both."
            )
        if not has_database_url and not has_full_host_config:
            raise ValueError(
                "MemoryConfig: no Postgres connection configured — set database_url, "
                "or postgres_host/postgres_user/postgres_password/postgres_database."
            )

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
        database_url = os.environ.get("DATABASE_URL")
        postgres_host = os.environ.get("POSTGRES_HOST")
        postgres_port_raw = os.environ.get("POSTGRES_PORT")

        return cls(
            database_url=database_url,
            postgres_host=postgres_host,
            postgres_port=int(postgres_port_raw) if postgres_port_raw else None,
            postgres_user=os.environ.get("POSTGRES_USER"),
            postgres_password=os.environ.get("POSTGRES_PASSWORD"),
            postgres_database=os.environ.get("POSTGRES_DATABASE"),
            postgres_schema=os.environ["POSTGRES_SCHEMA"],
            redis_url=os.environ.get("REDIS_URL"),
            upstash_url=os.environ.get("UPSTASH_REDIS_REST_URL"),
            upstash_token=os.environ.get("UPSTASH_REDIS_REST_TOKEN"),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            aws_llm_access_key_id=os.environ.get("AWS_LLM_ACCESS_KEY_ID"),
            aws_llm_secret_access_key=os.environ.get("AWS_LLM_SECRET_ACCESS_KEY"),
            embedding_dim=int(os.environ.get("MEMORY_VERSE_AVNEESH_EMBEDDING_DIM", "1024")),
        )
