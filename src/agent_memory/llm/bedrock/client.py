"""Shared boto3 Bedrock Runtime client factory."""

from __future__ import annotations

from typing import Any

import boto3


def create_bedrock_client(region_name: str) -> Any:
    return boto3.client("bedrock-runtime", region_name=region_name)
