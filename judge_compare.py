"""Liquid vs Bedrock on the same small judgement: is this web result this company's jobs page?

For each company, Nimble searches its careers pages once. Both models then judge every result with the same prompt and
schema. The scorecard: how often they agree, where they disagree (printed for a human to read), and the time and tokens
each one spent. Bedrock is the reference, not the truth; the disagreements are what a person reads.

  LIQUID_URL=http://127.0.0.1:8090 python judge_compare.py companies.json [--limit 25] [--roles "software engineer"]

companies.json: [{"name": "Nebius", "domain": "nebius.com"}, ...]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import liquid
import llm
from sponsors import Log, Web

HERE = Path(__file__).resolve().parent
REASON_FIRST = False

SCHEMA = {
    "type": "object", "required": ["same_company", "shows_jobs"],
    "properties": {"same_company": {"type": "boolean"}, "shows_jobs": {"type": "boolean"}},
}
# Variant: a one-line reason written before the booleans (small models often judge better when they reason first).
SCHEMA_REASON = {
    "type": "object", "required": ["reason", "same_company", "shows_jobs"],
    "properties": {"reason": {"type": "string", "maxLength": 200}, "same_company": {"type": "boolean"},
                   "shows_jobs": {"type": "boolean"}},
}
RULES = ("You judge one web search result. same_company: the result is about THIS company (check the website; a "
         "namesake, a different company, or a page listing many companies is false). shows_jobs: the result shows "
         "open jobs or a specific job posting. Answer from the title, url and text only.")


def judge(fn, company: dict, hit: dict) -> tuple[dict | None, float, int]:
    user = (f"COMPANY: {company['name']} (website: {company['domain']})\n\n"
            f"RESULT:\n{json.dumps(hit, ensure_ascii=False)[:2500]}")
    t = time.monotonic()
    try:
        ans, usage = fn(RULES, user, SCHEMA_REASON if REASON_FIRST else SCHEMA, max_tokens=200 if REASON_FIRST else 120)
    except Exception:  # noqa: BLE001 - a schema miss is a result for the scorecard, not a crash
        return None, time.monotonic() - t, 0
    return ans, time.monotonic() - t, usage.get("input_tokens", 0) + usage.get("output_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("companies")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--roles", default="software engineer")
    ap.add_argument("--reason-first", action="store_true", help="ask for a one-line reason before the booleans")
    args = ap.parse_args()
    global REASON_FIRST
    REASON_FIRST = args.reason_first
    if not liquid.available():
        raise SystemExit("set LIQUID_URL to the running llama-server")
    companies = json.loads(Path(args.companies).read_text(encoding="utf-8"))[: args.limit]
    web, log = Web(), Log()
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    def search(c: dict) -> list[tuple[dict, dict]]:
        try:
            res = web.search(f"{c['name']} careers {args.roles}", max_results=4)
        except Exception:  # noqa: BLE001
            return []
        return [(c, {"title": r.get("title", ""), "url": r.get("url", ""), "text": (r.get("description") or "")[:1200]})
                for r in res]

    with ThreadPoolExecutor(max_workers=6) as pool:
        pairs = [p for batch in pool.map(search, companies) for p in batch]
    print(f"{len(pairs)} results from {len(companies)} companies")

    rows = []
    # Liquid runs one call at a time on a laptop CPU; Bedrock in parallel. Each is timed per call.
    liq = [judge(liquid.ask, c, h) for c, h in pairs]
    with ThreadPoolExecutor(max_workers=6) as pool:
        bed = list(pool.map(lambda p: judge(llm.ask, *p), pairs))
    for (c, h), (la, lt, ltok), (ba, bt, btok) in zip(pairs, liq, bed):
        rows.append({"company": c["name"], "url": h["url"], "title": h["title"],
                     "liquid": la, "bedrock": ba, "liquid_s": round(lt, 2), "bedrock_s": round(bt, 2),
                     "liquid_tokens": ltok, "bedrock_tokens": btok})

    both = [r for r in rows if r["liquid"] and r["bedrock"]]
    agree = [r for r in both if r["liquid"] == r["bedrock"]]
    keep = lambda a: bool(a["same_company"] and a["shows_jobs"])  # noqa: E731 - the filter's actual keep decision
    agree_keep = [r for r in both if keep(r["liquid"]) == keep(r["bedrock"])]
    card = {
        "run": run, "reason_first": REASON_FIRST, "results": len(rows),
        "liquid_schema_misses": sum(1 for r in rows if not r["liquid"]),
        "bedrock_schema_misses": sum(1 for r in rows if not r["bedrock"]),
        "agree_both_fields": f"{len(agree)}/{len(both)}",
        "agree_keep_decision": f"{len(agree_keep)}/{len(both)}",
        "liquid_median_s": sorted(r["liquid_s"] for r in rows)[len(rows) // 2] if rows else 0,
        "bedrock_median_s": sorted(r["bedrock_s"] for r in rows)[len(rows) // 2] if rows else 0,
        "liquid_tokens": sum(r["liquid_tokens"] for r in rows),
        "bedrock_tokens": sum(r["bedrock_tokens"] for r in rows),
        "bedrock_model": llm.MODEL,
    }
    out = HERE / f"judge_compare_{run}.json"
    out.write_text(json.dumps({"scorecard": card, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        log.write([{"run": run, "ts": datetime.now(timezone.utc).isoformat(), "tool": "compare", "kind": "scorecard",
                    "text": json.dumps(card)[:500]}])
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(card, indent=1))
    print("\nDISAGREEMENTS on the keep decision (read these):")
    for r in both:
        if keep(r["liquid"]) != keep(r["bedrock"]):
            print(f"- {r['company']}: liquid={'keep' if keep(r['liquid']) else 'drop'} "
                  f"bedrock={'keep' if keep(r['bedrock']) else 'drop'} | {r['title'][:70]} | {r['url'][:90]}")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
