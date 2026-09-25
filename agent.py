"""The follow-up agent: plan, act, observe, self-correct, without re-reading its own history.

State it keeps (and is the only thing it reads to act):
  - the event card: title plus what happened there (talks, hosts), read once from the web
  - one lead card per person: who they are, why they matter, and 2-3 sourced facts found on the web
  - the style card: a short list of rules learned from the reader's swipes and fixes

History it does not read: every search, draft, swipe and fix is appended to Rawtree (the log) and to the feed the page
shows, but no model call ever gets the log. Each draft reads: style card + event card + one lead card. That is why the
context per draft stays flat while the log grows.

Code checks only exact things: the draft has no em dash and no link. Everything that needs judgement (which facts matter,
what a rejection means, how to rewrite) is a model call.
"""
from __future__ import annotations

import json

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import llm
from sponsors import Log, Web

HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"

CLOSING = "If it's relevant, I'd like to catch up."

START_STYLE = [
    "Plain text, no links in a first message",
    "Open with something they said, built or presented",
    "Subject for a job lead: <the actual role title> // <the company>",
    "End with: If it's relevant, I'd like to catch up.",
]


@dataclass
class Card:
    id: str
    lead_id: str
    channel: str
    subject: str
    body: str
    why: str
    sources: list[dict]
    status: str = "pending"      # pending | kept | fix
    version: int = 1
    rule_applied: str = ""


@dataclass
class State:
    reader: str = ""
    event: dict = field(default_factory=dict)
    leads: list[dict] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    style: list[str] = field(default_factory=lambda: list(START_STYLE))
    new_rule: str = ""
    step: str = "plan"
    feed: list[dict] = field(default_factory=list)
    events_logged: int = 0
    context_chars: list[int] = field(default_factory=list)
    history_chars: list[int] = field(default_factory=list)
    context_series: list[int] = field(default_factory=list)

    def save(self) -> None:
        STATE.write_text(json.dumps({**asdict(self)}, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls) -> "State":
        if not STATE.exists():
            return cls()
        raw = json.loads(STATE.read_text(encoding="utf-8"))
        raw["cards"] = [Card(**c) for c in raw.get("cards", [])]
        return cls(**raw)


class Agent:
    def __init__(self, state: State, log: Log | None = None, web: Web | None = None):
        self.s = state
        self.log = log
        self.web = web
        self.run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # ---- the log (append-only, never read back by a model) ----
    def note(self, tool: str, kind: str, text: str, **extra) -> None:
        row = {"run": self.run, "ts": datetime.now(timezone.utc).isoformat(), "tool": tool, "kind": kind,
               "text": text[:500], **extra}
        self.s.feed = (self.s.feed + [{"tool": tool, "text": text[:160]}])[-8:]
        self.s.events_logged += 1
        # The chart: history is everything logged so far; context is what the last draft actually read.
        self.s.history_chars.append(self.s.history_chars[-1] + len(json.dumps(row)) if self.s.history_chars else len(json.dumps(row)))
        # One point per logged event on both lines: what the latest draft read, carried forward until the next draft.
        self.s.context_series.append(extra.get("context_chars") or (self.s.context_series[-1] if self.s.context_series else 0))
        if self.log:
            try:
                self.log.write([row])
            except Exception:  # noqa: BLE001 - the demo keeps going if the log is down; the feed still shows it
                pass

    # ---- plan ----
    def plan(self, leads: list[dict], event: dict, reader: str, top: int = 12) -> None:
        """The room is already ranked by Actintro; the plan is the top leads with a reason, plus the event card."""
        self.s.step, self.s.reader, self.s.event = "plan", reader, event
        self.s.leads = sorted(leads, key=lambda l: -float(l.get("score") or 0))[:top]
        self.note("Actintro", "plan", f"Picked the {len(self.s.leads)} most relevant leads from {len(leads)} guests.")
        self.s.save()

    # ---- act: research then draft ----
    RESEARCH_SCHEMA = {
        "type": "object", "required": ["facts"],
        "properties": {"facts": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "required": ["fact", "source_title", "url"],
            "properties": {"fact": {"type": "string"}, "source_title": {"type": "string"}, "url": {"type": "string"}}}}},
    }

    def research(self, lead: dict) -> list[dict]:
        if not self.web:
            return lead.get("facts", [])
        hits = []
        for q in (f"{lead['name']} {lead.get('company', '')}", f"{lead.get('company', '')} careers jobs"):
            try:
                hits += [{"title": h.get("title", ""), "url": h.get("url", ""), "text": (h.get("description") or "")[:1200]}
                         for h in self.web.search(q, max_results=3)]
            except Exception as exc:  # noqa: BLE001
                self.note("Nimble", "error", f"search failed for {lead['name']}: {type(exc).__name__}")
        if not hits:
            return []
        ans, _ = llm.ask(
            "From these web results, keep at most 3 facts that are clearly about THIS person or THIS company (the same "
            "company, not a namesake) and would make a follow-up specific: a talk they gave, something they built or "
            "wrote, an open role that fits the reader. Drop anything about other companies or general articles. One "
            "fact per url, and the url exactly as given. Fewer facts is fine; never invent.",
            f"PERSON: {lead['name']}, {lead.get('title', '')} at {lead.get('company', '')}\nREADER: {self.s.reader}\n\n"
            f"RESULTS:\n{json.dumps(hits, ensure_ascii=False)[:9000]}",
            self.RESEARCH_SCHEMA, max_tokens=700)
        urls = {h["url"] for h in hits}
        facts, seen = [], set()
        for f in ans["facts"]:  # provenance: only urls the search returned, each once
            if f["url"] in urls and f["url"] not in seen:
                seen.add(f["url"])
                facts.append(f)
        for f in facts:
            self.note("Nimble", "fact", f"{lead['name']}: {f['fact']}", url=f["url"])
        return facts

    DRAFT_SCHEMA = {
        "type": "object", "required": ["channel", "subject", "body", "why"],
        "properties": {"channel": {"type": "string", "enum": ["Email", "LinkedIn note", "DM"]},
                       "subject": {"type": "string"}, "body": {"type": "string"}, "why": {"type": "string"}},
    }

    def _draft(self, lead: dict, extra: str = "") -> tuple[dict, int]:
        system = ("Write one follow-up from the reader to this person, after an event they both attended but did not "
                  "talk at. Follow every rule on the style card exactly. Use only facts on the lead card and the event "
                  "card; never invent. Plain text. No em dashes. No links. Pick the channel: Email when the lead card "
                  "has an email or a hiring angle, else LinkedIn note. `why` is one line on why this person, for the "
                  "reader. End the body with exactly this line and nothing after it, no sign-off: " + CLOSING +
                  (f"\n\nFIX: {extra}" if extra else ""))
        context = (f"STYLE CARD\n- " + "\n- ".join(self.s.style) +
                   f"\n\nEVENT CARD\n{json.dumps(self.s.event, ensure_ascii=False)[:1500]}"
                   f"\n\nREADER\n{self.s.reader}"
                   f"\n\nLEAD CARD\n{json.dumps({k: lead.get(k) for k in ('name', 'title', 'company', 'why', 'facts')}, ensure_ascii=False)}")
        ans: dict = {}
        for _ in range(3):
            ans, _ = llm.ask(system, context, self.DRAFT_SCHEMA, max_tokens=900)
            text = ans["subject"] + " " + ans["body"]
            # Exact checks only (no-handrolling.md): an em dash, a link, or a body that does not end with the reader's
            # own closing line word for word sends the draft back. Nothing is patched in code.
            problems = []
            if "—" in text:
                problems.append("it has an em dash")
            if "http" in text:
                problems.append("it has a link")
            if not ans["body"].rstrip().endswith(CLOSING):
                problems.append(f"the body must end with exactly: {CLOSING} (nothing after it, no sign-off)")
            if not problems:
                return ans, len(system) + len(context)
            system += "\n\nRewrite: " + "; ".join(problems) + "."
        return ans, len(system) + len(context)

    def act(self) -> None:
        self.s.step = "act"
        self.s.save()

        def one(lead: dict) -> tuple[dict, list[dict]]:
            return lead, self.research(lead)

        with ThreadPoolExecutor(max_workers=4) as pool:
            researched = list(pool.map(one, self.s.leads))
        for lead, facts in researched:
            lead["facts"] = facts
            ans, size = self._draft(lead)
            self.s.context_chars.append(size)
            card = Card(id=uuid.uuid4().hex[:8], lead_id=lead["id"], channel=ans["channel"], subject=ans["subject"],
                        body=ans["body"], why=ans["why"],
                        sources=[{"title": f["source_title"], "url": f["url"]} for f in facts])
            self.s.cards.append(card)
            self.note("Bedrock", "draft", f"Wrote the {card.channel.lower()} to {lead['name']}.", context_chars=size)
            self.s.save()

    # ---- observe ----
    def swipe(self, card_id: str, keep: bool) -> None:
        self.s.step = "observe"
        card = next(c for c in self.s.cards if c.id == card_id)
        card.status = "kept" if keep else "fix"
        lead = self.lead(card.lead_id)
        self.note("Rawtree", "swipe", f"{'Kept' if keep else 'Sent back to fix'}: {lead['name']}.", card=card_id)
        self.s.save()

    # ---- self-correct ----
    RULE_SCHEMA = {
        "type": "object", "required": ["rule", "style"],
        "properties": {"rule": {"type": "string"}, "style": {"type": "array", "maxItems": 8, "items": {"type": "string"}}},
    }

    def fix(self, card_ids: list[str], instruction: str) -> None:
        """The reader's fix becomes a rule on the style card (the memory), then the cards are rewritten with it."""
        self.s.step = "self_correct"
        cards = [c for c in self.s.cards if c.id in card_ids]
        ans, _ = llm.ask(
            "The reader rejected these drafts and said how to fix them. Turn that into ONE short, general rule for all "
            "future drafts (not about one person), and return the updated style card: keep the existing rules, merge or "
            "replace any the new rule contradicts, at most 8 rules, each under 12 words.",
            f"STYLE CARD\n- " + "\n- ".join(self.s.style) + f"\n\nREADER SAID: {instruction}\n\nREJECTED:\n" +
            "\n---\n".join(f"{c.subject}\n{c.body}" for c in cards),
            self.RULE_SCHEMA, max_tokens=500)
        self.s.style, self.s.new_rule = ans["style"], ans["rule"]
        self.note("Bedrock", "rule", f"New rule on your style card: {ans['rule']}")
        for c in cards:
            lead = self.lead(c.lead_id)
            new, size = self._draft(lead, extra=instruction)
            self.s.context_chars.append(size)
            c.channel, c.subject, c.body, c.why = new["channel"], new["subject"], new["body"], new["why"]
            c.status, c.version, c.rule_applied = "pending", c.version + 1, ans["rule"]
            self.note("Bedrock", "rewrite", f"Rewrote {lead['name']} with the new rule.", context_chars=size)
        self.s.save()

    def lead(self, lead_id: str) -> dict:
        return next(l for l in self.s.leads if l["id"] == lead_id)
