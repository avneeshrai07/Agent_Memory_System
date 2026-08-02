"""Shared boto3 Bedrock Runtime client factory."""

from __future__ import annotations

from typing import Any

import boto3


def create_bedrock_client(
    region_name: str,
    *,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> Any:
    """If credentials aren't passed explicitly, boto3 falls back to its
    normal default chain (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars,
    ~/.aws/credentials, or an instance/task IAM role) — the common
    production pattern. Pass them explicitly when the credentials meant for
    Bedrock are scoped separately from whatever else reads the standard
    AWS_* env var names, which boto3 won't discover on its own.
    """

    kwargs: dict[str, Any] = {"region_name": region_name}
    if aws_access_key_id is not None:
        kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key is not None:
        kwargs["aws_secret_access_key"] = aws_secret_access_key

    return boto3.client("bedrock-runtime", **kwargs)
