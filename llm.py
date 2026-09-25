"""One structured call to a model on Amazon Bedrock: the model must answer through a single tool whose input is our
JSON schema, and the answer is checked against the schema's required keys before it is used.

Standalone on purpose (boto3 only), so this folder runs as its own public repo.
"""
from __future__ import annotations

import os

import boto3
from botocore.config import Config

MODEL = os.environ.get("FOLLOWUPS_MODEL", "moonshotai.kimi-k2.5")
REGION = os.environ.get("AWS_REGION", "us-west-2")
_runtime = None


def _client():
    global _runtime
    if _runtime is None:
        _runtime = boto3.client("bedrock-runtime", region_name=REGION,
                                config=Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=180))
    return _runtime


def ask(system: str, user: str, schema: dict, *, max_tokens: int = 1500, tries: int = 2) -> tuple[dict, dict]:
    """Returns (answer, usage). Raises ValueError when the model misses the schema `tries` times."""
    last = None
    for _ in range(tries):
        raw = _client().converse(
            modelId=MODEL,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.4},
            toolConfig={"tools": [{"toolSpec": {"name": "answer", "description": "Your answer.",
                                                "inputSchema": {"json": schema}}}],
                        "toolChoice": {"tool": {"name": "answer"}}},
        )
        blocks = raw.get("output", {}).get("message", {}).get("content", [])
        answer = next((b["toolUse"]["input"] for b in blocks if "toolUse" in b), None)
        usage = raw.get("usage", {})
        if isinstance(answer, dict) and all(k in answer for k in schema.get("required", [])):
            return answer, {"input_tokens": usage.get("inputTokens", 0), "output_tokens": usage.get("outputTokens", 0)}
        last = answer
    raise ValueError(f"model missed the schema: {str(last)[:200]}")
