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
2. **Act**: search the web for each lead's company careers pages and open roles with **Nimble**. A model
   on **AWS Bedrock** keeps only the results about this company (not a namesake), then writes the follow-up from the
   facts that survived.
3. **Observe**: you swipe. Each swipe is logged.
4. **Self-correct**: a fix like "Shorter" becomes a general rule on the style card, the rejected drafts are rewritten,
   and every later draft follows the rule without being told again.

Checks in code are exact only: a draft with an em dash, a link, or a closing line that is not the reader's own is sent
back to the model, never patched. Everything that needs judgement (which facts matter, what a rejection means, how to
rewrite) is a model call.

## Sponsor tools

- **Tinybird (Rawtree)**: the append-only event log behind the page's history line and the audit trail.
- **Nimble**: live web search. Finds the event's talks and each lead's company careers pages and open roles. The agent
  works from the search results; it does not yet open each page, so it cannot tell a live posting from one that was
  taken down, and job boards can outrank the company's own careers page. Next: search the company's own domain and
  job board first, then open each source (Nimble extract) and keep only postings that are still open.
- **Black Forest Labs (FLUX 1.1 [pro])**: the faces of the made-up people in the sample room (`faces.py`). The names are
  invented, so no real person's photo can stand in; each face is generated once from the made-up name and role and
  saved in `faces/`, and the page says they are AI-generated.
- **AWS (Bedrock)**: filters the research, writes, turns fixes into rules and rewrites (infrastructure, not a
  hackathon sponsor).

### Tested today and left out: Liquid AI

We wired **LFM2.5-1.2B-Instruct** (local, llama.cpp) in as the research filter and then measured it against Bedrock on
the same 100 Nimble results from the real room (`judge_compare.py`): is this result this company's jobs page?

| | Liquid 1.2B (local) | Bedrock (Kimi K2.5) |
|---|---|---|
| Valid JSON | 100/100 | 100/100 |
| Median time per call | 2.0s | 3.8s |
| Said "same company" | 1/100 | 68/100 |
| Agreed on keep/drop | 47/100 | reference |

It called jobs.apple.com "not Apple" and metacareers.com "not Meta". Asking for a reason first made it worse (13/40),
and its reasons contradicted its answers. Fast and schema-perfect, but not fit for "whose page is this". The code path
stays (`liquid.py`, set `LIQUID_URL`) so the test can be rerun on a larger model.

A second test (`claim_check_test.py`) asked it a simpler question on ten draft openings: does this message claim the
recipient gave a talk? As asked, it said yes to all ten (5/10). With two worked examples and yes/no answers it caught
all five invented claims but raised two false alarms (8/10); Bedrock got 10/10. The larger LFM2.5-8B-A1B is a
reasoning model and was not tested in time.

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
- `liquid.py`: one structured call to a local Liquid model (off unless `LIQUID_URL` is set)
- `judge_compare.py`, `claim_check_test.py`: the Liquid vs Bedrock tests above
- `faces.py`: generates the sample room's faces with Black Forest Labs FLUX
- `sponsors.py`: Rawtree log and Nimble search/extract clients
- `server.py`: the API the page polls
- `index.html`: the page (product on the left, what the agent is doing on the right)
- `sample_room.json`: an invented room (made-up people at real companies)
