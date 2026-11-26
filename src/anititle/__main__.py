"""python -m anititle "<release title>" [--published-at 2026-07-05T13:06:52Z]"""

import argparse
import json

from anititle import agent


def main():
    parser = argparse.ArgumentParser(prog="anititle", description="Match a release title to an AniList entry and episode.")
    parser.add_argument("title")
    parser.add_argument("--published-at", help="publish time, ISO 8601")
    parser.add_argument("--model", default=agent.DEFAULT_MODEL)
    args = parser.parse_args()
    result = agent.match(args.title, args.published_at, model=args.model)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
