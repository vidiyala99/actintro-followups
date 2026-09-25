# Actintro Follow-ups

**Missed the people you meant to meet at last night's event?** This agent reads the room, researches your best leads
on the web, and writes the follow-ups. You just swipe: **Keep** (to your drafts) or **Fix** (and say how, in one tap).
It learns your style as you go, without re-reading everything you ever said.

Built at the **Long Horizon Agents Hack** (Tokens&, San Francisco, 2026-09-25).

## What is new today, and what already existed

- **Built today:** the follow-up agent, its memory (state cards), the fix-to-rule loop, the Rawtree event log and the
  live page.
- **Already existed:** [Actintro](https://actintro.com) (private) imports your Luma events, researches every guest and
  ranks the room against your goal. This agent plugs in after the event, as the "act on it" layer. The sample room here
  is invented; in Actintro it runs on a real ranked room.

## The long-horizon idea: preserve what matters, drop the rest

Long-running agents slow down and get less reliable because their history keeps growing. This agent never reads its
history. For every draft it reads exactly three small, explicit pieces of state:

| State | What it holds | How it changes |
|---|---|---|
| **Event card** | What happened at the event (talks, hosts), read once from the web | Written once |
| **Lead card** (one per person) | Who they are, why they matter, 2 to 3 sourced facts from the web | Written by research |
| **Style card** | Short rules learned from your fixes, e.g. "Keep first messages under 75 words" | Edited by the agent on every fix |

Everything that happens (each search, draft, swipe, fix, new rule) is appended to an event log in **Rawtree** and shown
on the page, but no model call ever receives the log. So the chart on the page shows history growing while the context
per draft stays flat.

## Plan, act, observe, self-correct

1. **Plan**: pick the most relevant leads from the ranked room.
2. **Act**: research each lead on the web with **Nimble** (their company's careers page, their talks and work), then
   write the follow-up with a model on **AWS Bedrock**.
3. **Observe**: you swipe. Each swipe is logged.
4. **Self-correct**: a fix like "Shorter" becomes a general rule on the style card, the rejected drafts are rewritten,
   and every later draft follows the rule without being told again.

Checks in code are exact only: a draft with an em dash, a link, or a closing line that is not the reader's own is sent
back to the model, never patched. Everything that needs judgement (which facts matter, what a rejection means, how to
rewrite) is a model call.

## Sponsor tools

- **Tinybird (Rawtree)**: the append-only event log behind the page's history line and the audit trail.
- **Nimble**: live web data. Reads the event page, each lead's company careers page, and their public work.
- **AWS (Bedrock)**: the model that researches, writes, turns fixes into rules and rewrites.

## Run it

```
pip install -r requirements.txt
cp .env.example .env        # add your Rawtree and Nimble keys; AWS credentials as usual
python server.py            # or: python server.py your_room.json
```

Open http://localhost:8765, press **Show me**, then swipe with the arrow keys or the buttons.

## Files

- `agent.py`: plan, research, draft, swipe, fix; the state cards and the exact checks
- `llm.py`: one structured call to Bedrock
- `sponsors.py`: Rawtree log and Nimble search/extract clients
- `server.py`: the API the page polls
- `index.html`: the page (product on the left, what the agent is doing on the right)
- `sample_room.json`: an invented room (made-up people at real companies)
