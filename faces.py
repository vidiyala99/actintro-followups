"""Faces for the made-up people in the sample room, generated once with Black Forest Labs FLUX and saved in the repo.

The demo names are invented, so no real person's photo can stand in for them. Each face is generated from the lead's
made-up name and role, written to faces/<lead id>.jpg, and the room file points at it. Run once; the page never waits
on it.

  BFL_API_KEY=... python faces.py [sample_room.json] [--model flux-pro-1.1] [--force]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sponsors import env

HERE = Path(__file__).resolve().parent
BASE = "https://api.bfl.ai/v1"


def _req(url: str, key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"x-key": key, "Content-Type": "application/json", "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.loads(res.read())


def generate(key: str, model: str, prompt: str) -> bytes:
    job = _req(f"{BASE}/{model}", key, {"prompt": prompt, "width": 512, "height": 512, "output_format": "jpeg"})
    for _ in range(120):
        time.sleep(1)
        res = _req(job["polling_url"], key)
        if res.get("status") == "Ready":
            # The signed image url is valid for about 10 minutes: download it now.
            with urllib.request.urlopen(res["result"]["sample"], timeout=60) as img:
                return img.read()
        if res.get("status") in ("Error", "Failed", "Content Moderated", "Request Moderated"):
            raise RuntimeError(f"BFL {res.get('status')}")
    raise TimeoutError("BFL did not finish in 2 minutes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("room", nargs="?", default=str(HERE / "sample_room.json"))
    ap.add_argument("--model", default="flux-pro-1.1")
    ap.add_argument("--force", action="store_true", help="regenerate faces that already exist")
    args = ap.parse_args()
    key = env("BFL_API_KEY")
    room_path = Path(args.room)
    room = json.loads(room_path.read_text(encoding="utf-8"))
    (HERE / "faces").mkdir(exist_ok=True)

    def one(lead: dict) -> tuple[str, str]:
        out = HERE / "faces" / f"{lead['id']}.jpg"
        if out.exists() and not args.force:
            return lead["id"], "kept existing"
        prompt = (f"Professional headshot photo of a fictional person named {lead['name']}, "
                  f"{lead.get('title', '')} at a tech company, at a developer meetup in San Francisco. Head and "
                  "shoulders, looking at the camera, relaxed natural expression, soft window light, plain warm grey "
                  "background, 85mm lens, shallow depth of field, realistic skin, no text, no logos.")
        try:
            out.write_bytes(generate(key, args.model, prompt))
            return lead["id"], "generated"
        except Exception as exc:  # noqa: BLE001 - one failed face leaves that card on initials
            return lead["id"], f"failed: {type(exc).__name__}: {str(exc)[:80]}"

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = dict(pool.map(one, room["leads"]))
    for lead in room["leads"]:
        if (HERE / "faces" / f"{lead['id']}.jpg").exists():
            lead["photo"] = f"faces/{lead['id']}.jpg"
        print(f"{lead['id']} {lead['name']}: {results[lead['id']]}")
    room["faces_note"] = f"Faces are AI-generated with Black Forest Labs {args.model}; the people do not exist."
    room_path.write_text(json.dumps(room, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
