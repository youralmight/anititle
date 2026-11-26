"""AniList GraphQL client: search, one entry with its relations, many entries by id."""

import hashlib
import json
import time
from pathlib import Path

import httpx

_URL = "https://graphql.anilist.co"
# AniList allows 30 requests a minute per IP and every agent runs its own
# tool process, so answers are shared through a disk cache.
_CACHE = Path.home() / ".cache" / "anititle" / "anilist"
_TTL = 12 * 3600

_FIELDS = """
id
title { romaji english native }
synonyms
format
status
episodes
season
seasonYear
startDate { year month day }
endDate { year month day }
"""

_SEARCH = f"""
query ($search: String, $perPage: Int) {{
  Page(page: 1, perPage: $perPage) {{
    media(search: $search, type: ANIME) {{ {_FIELDS} }}
  }}
}}
"""

_BY_IDS = f"""
query ($ids: [Int]) {{
  Page(page: 1, perPage: 50) {{
    media(id_in: $ids, type: ANIME) {{ {_FIELDS} }}
  }}
}}
"""

_WITH_RELATIONS = f"""
query ($ids: [Int]) {{
  Page(page: 1, perPage: 50) {{
    media(id_in: $ids, type: ANIME) {{
      {_FIELDS}
      relations {{ edges {{ relationType node {{ id type }} }} }}
    }}
  }}
}}
"""

_ONE = f"""
query ($id: Int) {{
  Media(id: $id, type: ANIME) {{
    {_FIELDS}
    nextAiringEpisode {{ episode airingAt }}
    relations {{ edges {{ relationType node {{ type {_FIELDS} }} }} }}
  }}
}}
"""


def _post(query: str, variables: dict) -> dict:
    key = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
    path = _CACHE / f"{key}.json"
    if path.exists() and time.time() - path.stat().st_mtime < _TTL:
        return json.loads(path.read_text())
    data = _fetch(query, variables)
    _CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    return data


def _fetch(query: str, variables: dict) -> dict:
    """POST with up to 3 tries; on 429, wait as long as Retry-After says."""
    for attempt in range(3):
        response = httpx.post(_URL, json={"query": query, "variables": variables}, timeout=15)
        if response.status_code == 429 and attempt < 2:
            time.sleep(float(response.headers.get("retry-after", 5)))
            continue
        response.raise_for_status()
        return response.json()["data"]
    raise RuntimeError("unreachable")


def search(text: str, limit: int = 8) -> list[dict]:
    return _post(_SEARCH, {"search": text, "perPage": limit})["Page"]["media"]


def by_ids(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    return _post(_BY_IDS, {"ids": ids[:50]})["Page"]["media"]


def with_relations(ids: list[int]) -> list[dict]:
    """Entries by id (up to 50), each with its relation edges (related ids only)."""
    if not ids:
        return []
    return _post(_WITH_RELATIONS, {"ids": sorted(ids)[:50]})["Page"]["media"]


def one(anilist_id: int) -> dict | None:
    return _post(_ONE, {"id": anilist_id})["Media"]


_LAST_AIRED = """
query ($id: Int, $before: Int) {
  Page(page: 1, perPage: 1) {
    airingSchedules(mediaId: $id, airingAt_lesser: $before, sort: EPISODE_DESC) { episode }
  }
}
"""


def last_aired(anilist_id: int, before: int) -> int | None:
    """The highest episode that aired before unix time `before`, None without schedule data."""
    schedules = _post(_LAST_AIRED, {"id": anilist_id, "before": before})["Page"]["airingSchedules"]
    return schedules[0]["episode"] if schedules else None
