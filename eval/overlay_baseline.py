"""Run the same dataset through anilist-overlay's fixed pipeline (/anime/identify).

    OVERLAY_TOKEN=... uv run python eval/overlay_baseline.py [--url http://127.0.0.1:20211]

The overlay API listens on the homeserver's loopback; forward it first:
    ssh -N -L 20211:127.0.0.1:20211 homeserver

Overlay results are mapped onto ours (non_anime -> not_anime, pending_review ->
error without a type) and scored with the same function as eval/run.py.

An expected answer is what a careful person concludes from the exact input. The
overlay has no "no publish date" mode, so undated titles are sent with today's
date; for the current undated items that date does not change the answer.
"""

import argparse
import datetime
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx

from run import HERE, score

ATTEMPTS = 4
RETRY_WAIT = 30


def to_ours(overlay: dict) -> dict:
    status = overlay.get("status")
    if status == "matched":
        return {"status": "matched", "anilist_id": overlay["anime_id"], "episode": overlay.get("episode"),
                "error_type": None}
    error_type = {"non_anime": "not_anime"}.get(status)
    return {"status": "error", "anilist_id": None, "episode": None, "error_type": error_type}


def identify(url: str, headers: dict, title: str, day: str) -> tuple[dict, float]:
    """One answer and the seconds it took. A service failure is not an answer:
    retry it, and stop the whole run if it persists rather than score it."""
    for attempt in range(ATTEMPTS):
        start = time.monotonic()
        try:
            response = httpx.post(f"{url}/anime/identify", headers=headers, timeout=180,
                                  json={"title": title, "today": day})
            response.raise_for_status()
            # Without ?trace=1 the endpoint returns identify_anime's (result, trace) tuple as a list.
            body = response.json()
            overlay = body[0] if isinstance(body, list) else body
            if overlay.get("status") != "error":
                return overlay, round(time.monotonic() - start, 1)
            failure = overlay
        except httpx.HTTPError as e:
            failure = repr(e)
        print(f"service failure, attempt {attempt + 1}/{ATTEMPTS}: {failure}", flush=True)
        time.sleep(RETRY_WAIT)
    raise SystemExit(f"overlay still failing on {title!r}; run aborted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:20211")
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {os.environ['OVERLAY_TOKEN']}"}
    today = datetime.date.today().isoformat()

    items = [json.loads(line) for line in open(HERE / "dataset.jsonl")]
    out_path = HERE / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-overlay.jsonl"
    rows = []
    with open(out_path, "w") as out:
        for item in items:
            overlay, seconds = identify(args.url, headers, item["title"], (item["published_at"] or today)[:10])
            got = {**to_ours(overlay), "seconds": seconds, "overlay": overlay}
            row = {"id": item["id"], "difficulty": item["difficulty"], "tags": item["tags"],
                   "expected": item["expected"], "got": got, "verdict": score(item["expected"], got)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(f"{row['id']} {row['verdict']:<24} {got['seconds']:>5}s  {overlay}", flush=True)

    print(f"\n{out_path}")
    for difficulty in ("easy", "hard"):
        print(difficulty, dict(Counter(r["verdict"] for r in rows if r["difficulty"] == difficulty)))
    seconds = [r["got"]["seconds"] for r in rows]
    print(f"seconds: mean {sum(seconds) / len(seconds):.1f}, max {max(seconds):.1f}")


if __name__ == "__main__":
    main()
