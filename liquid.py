"""Liquid AI's LFM2.5-1.2B-Instruct, run locally through llama.cpp's OpenAI-compatible server.

The agent sends Liquid the small, frequent judgement calls (is this search result about THIS company, which facts are
worth keeping), and keeps Bedrock for writing. Answers are constrained to our JSON schema by the server.

  llama-server -m LFM2.5-1.2B-Instruct-Q4_K_M.gguf --port 8090 -c 8192
  LIQUID_URL=http://127.0.0.1:8090   (unset: the agent uses Bedrock for these calls too)
"""
from __future__ import annotations

import json
import urllib.request

from sponsors import env

URL = env("LIQUID_URL", default="")


def available() -> bool:
    return bool(URL)


def ask(system: str, user: str, schema: dict, *, max_tokens: int = 700) -> tuple[dict, dict]:
    body = {
        "model": "LFM2.5-1.2B-Instruct",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
    }
    req = urllib.request.Request(f"{URL.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as res:
        data = json.loads(res.read())
    answer = json.loads(data["choices"][0]["message"]["content"])
    if not all(k in answer for k in schema.get("required", [])):
        raise ValueError("liquid missed the schema")
    usage = data.get("usage", {})
    return answer, {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
