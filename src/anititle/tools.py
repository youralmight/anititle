"""What the agent can look up, as compact JSON-ready dicts.

Each result carries the facts the matching rules need (episode count, air
dates, format) so a typical title is settled in two or three calls.
"""

from datetime import date, datetime, timedelta, timezone
from functools import cache

from anititle import anilist, chinese

# Formats whose episodes are numbered as part of a TV run; used for the
# cross-season episode offset.
_SERIES_FORMATS = {"TV", "TV_SHORT", "ONA"}


def _date(d: dict | None) -> str | None:
    if not d or not d.get("year"):
        return None
    return "-".join(f"{v:02d}" for v in (d["year"], d.get("month"), d.get("day")) if v)


def summary(media: dict) -> dict:
    title = media.get("title") or {}
    titles = [title.get("romaji"), title.get("english"), title.get("native")]
    titles += chinese.names().get(media["id"], [])[:4] + (media.get("synonyms") or [])[:3]
    return {
        "id": media["id"],
        "titles": list(dict.fromkeys(t for t in titles if t)),
        "format": media.get("format"),
        "status": media.get("status"),
        "episodes": media.get("episodes"),
        "start": _date(media.get("startDate")),
        "end": _date(media.get("endDate")),
    }


@cache
def _one(anilist_id: int) -> dict | None:
    return anilist.one(anilist_id)


def search_anime(query: str) -> list[dict]:
    """AniList's own search plus the Chinese-name index, merged, Chinese hits first."""
    chinese_ids = chinese.search(query)
    found = {m["id"]: m for m in anilist.by_ids(chinese_ids)}
    for media in anilist.search(query):
        found.setdefault(media["id"], media)
    order = chinese_ids + [i for i in found if i not in chinese_ids]
    return [summary(found[i]) for i in order if i in found]


def get_anime(anilist_id: int) -> dict:
    """One entry with its next airing episode and its anime relations."""
    media = _one(anilist_id)
    if media is None:
        return {"error": f"no anime with id {anilist_id}"}
    relations = [
        {"relation": edge["relationType"], **summary(edge["node"])}
        for edge in media["relations"]["edges"]
        if edge["node"].get("type") == "ANIME"
    ]
    return {**summary(media), "next_airing_episode": media.get("nextAiringEpisode"), "relations": relations}


# AniList start/end dates are Japanese broadcast dates.
_JST = timezone(timedelta(hours=9))


def _parse_moment(text: str) -> datetime:
    """A publish time as an aware datetime. A bare date means the end of that day in Japan;
    a time without an offset is taken as UTC."""
    if len(text) == 10:
        return datetime.fromisoformat(text).replace(hour=23, minute=59, second=59, tzinfo=_JST)
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _aired_by(media: dict, moment: datetime) -> int | None:
    """Episodes of `media` aired by `moment`."""
    day = moment.astimezone(_JST).date()
    start, end = _date(media.get("startDate")), _date(media.get("endDate"))
    if start and len(start) == 10 and day < date.fromisoformat(start):
        return 0
    if end and len(end) == 10 and day > date.fromisoformat(end):
        return media.get("episodes")
    return anilist.last_aired(media["id"], int(moment.timestamp()))


def _start_key(media: dict) -> tuple:
    d = media.get("startDate") or {}
    return (d.get("year") is None, d.get("year") or 0, d.get("month") or 0, d.get("day") or 0)


def _franchise(anilist_id: int, max_entries: int) -> list[dict]:
    """Entries linked to this one through PREQUEL/SEQUEL edges in either direction,
    in air-date order. AniList's edges are not always consistent (a later season
    can be marked as an earlier one's prequel), so the order comes from start dates,
    not from following the edges."""
    found: dict[int, dict] = {}
    frontier = [anilist_id]
    while frontier and len(found) < max_entries:
        layer = anilist.with_relations(frontier[: max_entries - len(found)])
        frontier = []
        for media in layer:
            found[media["id"]] = media
        for media in layer:
            for edge in media["relations"]["edges"]:
                node = edge["node"]
                if (edge["relationType"] in ("PREQUEL", "SEQUEL") and node.get("type") == "ANIME"
                        and node["id"] not in found and node["id"] not in frontier):
                    frontier.append(node["id"])
    return sorted(found.values(), key=_start_key)


def season_chain(anilist_id: int, on_date: str | None = None, max_entries: int = 25) -> list[dict]:
    """The franchise run this entry belongs to (PREQUEL/SEQUEL links), oldest first.

    Every entry gets `episodes_before`: the TV/ONA episodes of earlier entries,
    i.e. the offset a continuously numbered release would add. With `on_date`
    (the publish time, ISO 8601) every entry also gets `aired_by_date`: its last
    episode aired by then.
    """
    chain = _franchise(anilist_id, max_entries)
    if not chain:
        return [{"error": f"no anime with id {anilist_id}"}]
    result, offset = [], 0
    moment = _parse_moment(on_date) if on_date else None
    for entry in chain:
        counted = entry.get("format") in _SERIES_FORMATS
        row = {**summary(entry), "episodes_before": offset if counted else None}
        if moment:
            row["aired_by_date"] = _aired_by(entry, moment)
        result.append(row)
        if counted:
            offset += entry.get("episodes") or 0
    return result
