"""soruly/anilist-chinese: community-maintained Chinese names for AniList ids.

The JSON is cached on disk for a week; search is a substring match after
both sides are converted to Simplified Chinese with punctuation removed.
"""

import json
import time
import unicodedata
from functools import cache
from pathlib import Path

import httpx
import opencc

_URL = "https://raw.githubusercontent.com/soruly/anilist-chinese/refs/heads/master/anilist-chinese.json"
_CACHE = Path.home() / ".cache" / "anititle" / "anilist-chinese.json"
_TTL = 7 * 24 * 3600

_t2s = opencc.OpenCC("t2s")


def normalize(text: str) -> str:
    """Simplified Chinese, lower case, without punctuation, symbols or spaces.

    Spacing between Latin and CJK text varies freely ("LV999的村民" vs
    "LV999 的村民"), so whitespace is dropped rather than collapsed."""
    text = _t2s.convert(text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZ" and not ch.isspace())


def _download() -> list:
    if not _CACHE.exists() or time.time() - _CACHE.stat().st_mtime > _TTL:
        response = httpx.get(_URL, timeout=60, follow_redirects=True)
        response.raise_for_status()
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_bytes(response.content)
    return json.loads(_CACHE.read_text())


@cache
def names() -> dict[int, list[str]]:
    """AniList id -> its Chinese title followed by its Chinese synonyms."""
    result = {}
    for entry in _download():
        title = entry.get("title") or ""
        if isinstance(title, dict):
            title = title.get("native") or title.get("romaji") or ""
        synonyms = entry.get("synonyms") or []
        if entry.get("id"):
            result[entry["id"]] = [n for n in [str(title), *map(str, synonyms)] if n]
    return result


@cache
def _index() -> list[tuple[str, int]]:
    return [(n, i) for i, ns in names().items() for n in map(normalize, ns) if n]


def search(text: str, limit: int = 8) -> list[int]:
    """Ids whose Chinese name contains `text`, shortest name first (closest match).

    With no such name, falls back to names contained in `text`, longest first:
    a release often wraps the name in extra words ("完全相反的你和我").
    """
    query = normalize(text)
    if not query:
        return []
    hits: dict[int, int] = {}
    for name, anilist_id in _index():
        if query in name:
            hits[anilist_id] = min(len(name), hits.get(anilist_id, 1 << 30))
    if hits:
        return sorted(hits, key=hits.get)[:limit]
    for name, anilist_id in _index():
        if len(name) >= 3 and name in query:
            hits[anilist_id] = max(len(name), hits.get(anilist_id, 0))
    return sorted(hits, key=hits.get, reverse=True)[:limit]
