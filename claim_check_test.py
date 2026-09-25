"""Can Liquid (local LFM2.5-1.2B) catch a draft that claims the recipient gave a talk they did not give?

Ten draft openings with known answers: five say the reader saw the recipient present (the first is a line the agent
actually wrote today for a made-up person), five do not (a named speaker's talk, general demos, a job, their company).
Liquid and Bedrock get the same question and schema.

  LIQUID_URL=http://127.0.0.1:8090 python claim_check_test.py
"""
from __future__ import annotations

import time

import liquid
import llm

SCHEMA = {"type": "object", "required": ["claims_recipient_presented"],
          "properties": {"claims_recipient_presented": {"type": "boolean"}}}
RULES = ("You check one outreach message written TO a person. claims_recipient_presented is true only when the message "
         "says or implies that the RECIPIENT personally gave a talk, demo, presentation or post (for example 'your "
         "talk', 'you presented', 'great demo yesterday'). Talks by other named people, demos in general, the "
         "recipient's company or job openings are false.")
CASES = [
    (True, "Hi Jordan, I caught your talk at the Agents & APIs meetup. The discussion on building agentic applications with Firebase was particularly relevant to my work."),
    (True, "Hi Marcus, loved your demo on Postman Collections as guardrails last night."),
    (True, "Hi Sofia, your presentation on agent evals at the meetup stuck with me."),
    (True, "Hi Nina, I saw you present the Firebase Agent Skills demo last week and wanted to follow up."),
    (True, "Hi Daniel, great talk yesterday on agentic brokerage APIs."),
    (False, "Hi Jordan, I was at the Agents & APIs meetup last night and caught Tristan Denyer's talk on using Postman Collections as guardrails for AI agents."),
    (False, "Hi Ethan, I was at the Agents & APIs meetup and caught the demos on agentic brokerage APIs and Firebase Agent Skills."),
    (False, "Hi Priya, I noticed Google Cloud is hiring a Senior Software Engineer, Backend."),
    (False, "Hi Leah, I was at the Postman meetup on agentic apps last week. The demos on Firebase Agent Skills stuck with me."),
    (False, "Hi Ethan, your work at Carmel Labs caught my attention as an early-stage founder in this space."),
]


def run(name: str, fn) -> None:
    right, times = 0, []
    for want, text in CASES:
        t = time.monotonic()
        try:
            got = fn(RULES, f"MESSAGE:\n{text}", SCHEMA, max_tokens=60)[0]["claims_recipient_presented"]
        except Exception as exc:  # noqa: BLE001 - a miss is a result
            got = f"error {type(exc).__name__}"
        times.append(time.monotonic() - t)
        right += got is want
        print(f"  {name:8} want={want!s:5} got={got!s:5} {'ok ' if got is want else 'MISS'} | {text[:70]}")
    print(f"{name}: {right}/{len(CASES)} right, median {sorted(times)[len(times) // 2]:.1f}s per call\n")


if __name__ == "__main__":
    if not liquid.available():
        raise SystemExit("set LIQUID_URL")
    run("Liquid", liquid.ask)
    run("Bedrock", llm.ask)
