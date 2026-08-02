import pytest

from memory_verse_avneesh.config import MemoryConfig


def test_valid_config_with_standard_redis():
    config = MemoryConfig(
        postgres_dsn="postgresql://x", postgres_schema="public",
        redis_url="redis://localhost",
    )
    assert config.redis_url == "redis://localhost"
    assert config.upstash_url is None


def test_valid_config_with_upstash():
    config = MemoryConfig(
        postgres_dsn="postgresql://x",
        postgres_schema="public",
        upstash_url="https://example.upstash.io",
        upstash_token="tok",
    )
    assert config.redis_url is None
    assert config.upstash_url == "https://example.upstash.io"


def test_raises_when_no_cache_backend_configured():
    with pytest.raises(ValueError, match="no Tier 0/1 cache backend"):
        MemoryConfig(postgres_dsn="postgresql://x", postgres_schema="public")


def test_raises_when_both_cache_backends_configured():
    with pytest.raises(ValueError, match="not both"):
        MemoryConfig(
            postgres_dsn="postgresql://x",
            postgres_schema="public",
            redis_url="redis://localhost",
            upstash_url="https://example.upstash.io",
            upstash_token="tok",
        )


def test_raises_when_only_upstash_url_set():
    with pytest.raises(ValueError, match="must both be set"):
        MemoryConfig(
            postgres_dsn="postgresql://x", postgres_schema="public",
            upstash_url="https://example.upstash.io",
        )


def test_raises_when_only_upstash_token_set():
    with pytest.raises(ValueError, match="must both be set"):
        MemoryConfig(
            postgres_dsn="postgresql://x", postgres_schema="public",
            upstash_token="tok",
        )


def test_missing_postgres_schema_raises_type_error():
    # postgres_schema has no default, deliberately -- a silent "public"
    # default risks unrelated apps colliding on the same table.
    with pytest.raises(TypeError):
        MemoryConfig(postgres_dsn="postgresql://x", redis_url="redis://localhost")


def test_from_env_reads_upstash_vars(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://x")
    monkeypatch.setenv("POSTGRES_SCHEMA", "public")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "tok")

    config = MemoryConfig.from_env()

    assert config.postgres_schema == "public"
    assert config.upstash_url == "https://example.upstash.io"
    assert config.upstash_token == "tok"
    assert config.redis_url is None


def test_from_env_reads_standard_redis_var(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://x")
    monkeypatch.setenv("POSTGRES_SCHEMA", "public")
    monkeypatch.setenv("REDIS_URL", "redis://localhost")
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)

    config = MemoryConfig.from_env()

    assert config.redis_url == "redis://localhost"
    assert config.upstash_url is None


def test_from_env_raises_when_postgres_schema_missing(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://x")
    monkeypatch.delenv("POSTGRES_SCHEMA", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://localhost")

    with pytest.raises(KeyError):
        MemoryConfig.from_env()
