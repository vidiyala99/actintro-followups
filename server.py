"""A small web server for the follow-up agent: one page, and a JSON API the page polls.

  python server.py [room.json]      then open http://localhost:8765

room.json: {"event": {"title", "url", "description"}, "reader": "who the reader is and what they want",
            "leads": [{"id", "name", "title", "company", "why", "score"}]}
sample_room.json is an invented room; a real room comes from Actintro's ranking export.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import llm
from agent import STATE, Agent, State
from sponsors import Log, Web

HERE = Path(__file__).resolve().parent
ROOM = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sample_room.json"
app = FastAPI()
lock = threading.Lock()
agent = Agent(State.load(), Log(), Web())
busy = {"on": False, "what": ""}


def background(what: str, fn) -> None:
    def run():
        busy.update(on=True, what=what)
        try:
            with lock:
                fn()
        except Exception as exc:  # noqa: BLE001 - shown on the page, never a crash
            agent.note("Agent", "error", f"{what} failed: {type(exc).__name__}: {str(exc)[:120]}")
            agent.s.save()
        finally:
            busy.update(on=False, what="")
    threading.Thread(target=run, daemon=True).start()


EVENT_SCHEMA = {"type": "object", "required": ["summary", "talks"],
                "properties": {"summary": {"type": "string"},
                               "talks": {"type": "array", "maxItems": 6, "items": {"type": "string"}}}}


def read_event(room: dict) -> dict:
    """The event card: what happened there, read once from the web (Nimble) and kept, never re-searched."""
    event = dict(room["event"])
    try:
        hits = agent.web.search(f"{event['title']} speakers talks", max_results=3)
        page = "\n\n".join((h.get("description") or "")[:3000] for h in hits)
        if page:
            ans, _ = llm.ask("Summarise this event for someone writing follow-ups: one line on what it was, and the "
                             "talks as 'Speaker (Company): topic'. Only what the text says.",
                             f"EVENT: {event['title']}\n\n{page[:8000]}", EVENT_SCHEMA, max_tokens=600)
            event.update(summary=ans["summary"], talks=ans["talks"])
            agent.note("Nimble", "event", f"Read the event page: {len(ans['talks'])} talks found.")
    except Exception as exc:  # noqa: BLE001
        agent.note("Nimble", "error", f"event page failed: {type(exc).__name__}")
    return event


class Swipe(BaseModel):
    id: str
    keep: bool


class Fix(BaseModel):
    ids: list[str]
    instruction: str


@app.get("/")
def page():
    return FileResponse(HERE / "index.html")


@app.get("/api/state")
def state():
    s = agent.s
    leads = {l["id"]: l for l in s.leads}
    cards = [{**c.__dict__, "lead": {k: leads.get(c.lead_id, {}).get(k) for k in ("name", "title", "company", "why")}}
             for c in s.cards]
    return {"step": s.step, "busy": busy, "event": s.event, "style": s.style, "new_rule": s.new_rule,
            "feed": s.feed, "cards": cards, "events_logged": s.events_logged,
            "context_chars": s.context_series[-60:], "history_chars": s.history_chars[-60:],
            "last_context": s.context_chars[-1] if s.context_chars else 0,
            "leads_total": len(s.leads)}


@app.post("/api/start")
def start():
    room = json.loads(ROOM.read_text(encoding="utf-8"))
    agent.s = State()
    STATE.unlink(missing_ok=True)

    def go():
        agent.s.event = read_event(room)
        agent.s.names_invented = bool(room.get("names_invented"))
        agent.s.target_roles = room.get("target_roles", "software engineer")
        agent.plan(room["leads"], agent.s.event, room["reader"], top=room.get("top", 12))
        agent.act()
    background("Writing follow-ups", go)
    return {"ok": True}


@app.post("/api/swipe")
def swipe(body: Swipe):
    with lock:
        agent.swipe(body.id, body.keep)
    return {"ok": True}


@app.post("/api/fix")
def fix(body: Fix):
    background("Fixing", lambda: agent.fix(body.ids, body.instruction))
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
