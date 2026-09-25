"""The two outside services the follow-through agent uses, as thin HTTP clients.

Rawtree (Tinybird): the append-only event log. Every thing that happens (a guest arrives, a reply lands, the agent edits
a state card, a draft is written) is one row, never edited. The agent does not read this log to think; it reads it only
for the dashboard and for audit. That split is the point: what persists lives here, what the agent needs lives on the
state card.

Nimble: live web data. Before the agent drafts a follow-up it re-checks one fact on the open web (is that role still
posted), because a state card written days ago can be stale.

Keys come from the environment or ./.env (TINYBIRD_TOKEN, TINYBIRD_HOST, NIMBLE_API_KEY) and are never printed.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Keys: environment first, then ./.env (never committed; see .env.example).
ENV_FILES = [HERE / ".env"]


def env(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for env_file in ENV_FILES:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit(f"missing {name}: set it in the environment or in .env")


def _post(url: str, token: str, body, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode() or "{}")


class Log:
    """Append-only rows in one Rawtree table."""

    def __init__(self, table: str = "followup_events"):
        self.table = table
        self.host = env("TINYBIRD_HOST").rstrip("/")
        self.token = env("TINYBIRD_TOKEN")

    def write(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        return int(_post(f"{self.host}/v1/tables/{self.table}", self.token, rows).get("inserted", 0))

    def query(self, sql: str) -> list[dict]:
        return _post(f"{self.host}/v1/query", self.token, {"sql": sql}).get("data", [])


class Web:
    """Nimble search and extract."""

    BASE = "https://sdk.nimbleway.com/v2"

    def __init__(self):
        self.key = env("NIMBLE_API_KEY")

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        out = _post(f"{self.BASE}/search", self.key,
                    {"query": query, "focus": "general", "max_results": max_results, "search_depth": "standard"})
        return out.get("results") or out.get("data") or []

    def extract(self, url: str) -> dict:
        return _post(f"{self.BASE}/extract", self.key, {"url": url, "formats": ["markdown"]}, timeout=120)
