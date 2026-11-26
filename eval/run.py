"""Run the agent over eval/dataset.jsonl and score it.

    uv run python eval/run.py [--only h01,h02] [--limit N] [--workers 2] [--model ...]

Writes one JSON line per title to eval/runs/<timestamp>.jsonl and prints a summary.
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anititle import agent

HERE = Path(__file__).parent


def score(expected: dict, got: dict) -> str:
    if expected["status"] == "matched":
        if got["status"] != "matched":
            return "false_error"  # should have answered, reported an error
        same = (got["anilist_id"], got["episode"]) == (expected["anilist_id"], expected["episode"])
        return "correct" if same else "wrong_answer"
    if got["status"] == "matched":
        return "forced_answer"  # should have reported an error, answered anyway
    return "correct" if got["error_type"] == expected["error_type"] else "correct_wrong_error_type"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated item ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default=agent.DEFAULT_MODEL)
    args = parser.parse_args()

    items = [json.loads(line) for line in open(HERE / "dataset.jsonl")]
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i["id"] in wanted]
    items = items[: args.limit]

    out_path = HERE / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    def run(item):
        got = agent.match(item["title"], item["published_at"], model=args.model).to_dict()
        row = {"id": item["id"], "difficulty": item["difficulty"], "tags": item["tags"],
               "expected": item["expected"], "got": got, "verdict": score(item["expected"], got)}
        print(f"{row['id']} {row['verdict']:<24} {got['seconds']:>5}s  {got['anilist_id']}/{got['episode']} {got['error_type'] or ''}",
              flush=True)
        return row

    with ThreadPoolExecutor(args.workers) as pool, open(out_path, "w") as out:
        rows = []
        for row in pool.map(run, items):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    seconds = [r["got"]["seconds"] for r in rows]
    print(f"\n{out_path}")
    for difficulty in ("easy", "hard"):
        sub = [r for r in rows if r["difficulty"] == difficulty]
        if sub:
            print(difficulty, dict(Counter(r["verdict"] for r in sub)))
    print(f"seconds: mean {sum(seconds) / len(seconds):.1f}, max {max(seconds):.1f}")
    usage = [r["got"]["usage"] for r in rows if r["got"]["usage"]]
    for key in ("model_calls", "input_tokens", "cache_read", "output_tokens"):
        if usage:
            print(f"{key}: mean {sum(u[key] for u in usage) / len(usage):.1f}")


if __name__ == "__main__":
    main()
