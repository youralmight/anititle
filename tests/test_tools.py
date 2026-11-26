"""Network-free tests: AniList and the Chinese-name data are replaced by small fixtures."""

import pytest

from anititle import chinese, tools


def media(id, fmt="TV", episodes=12, start=(2026, 1, 1), end=(2026, 3, 31), prequel=None, sequel=None, title=None):
    edges = []
    if prequel:
        edges.append({"relationType": "PREQUEL", "node": {"id": prequel, "type": "ANIME"}})
    if sequel:
        edges.append({"relationType": "SEQUEL", "node": {"id": sequel, "type": "ANIME"}})
    as_date = lambda d: dict(zip(("year", "month", "day"), d)) if d else {"year": None}
    return {"id": id, "title": {"romaji": title or f"show {id}"}, "synonyms": [], "format": fmt, "status": "FINISHED",
            "episodes": episodes, "startDate": as_date(start), "endDate": as_date(end),
            "relations": {"edges": edges}}


@pytest.fixture
def no_chinese_names(monkeypatch):
    monkeypatch.setattr(chinese, "names", lambda: {})


@pytest.fixture
def franchise(monkeypatch, no_chinese_names):
    """S1 (12 eps) -> movie -> S2 (13 eps, Jul-Sep 2026) -> S3 (airing from Jan 2027)."""
    entries = {
        1: media(1, sequel=2, start=(2026, 1, 1), end=(2026, 3, 31)),
        2: media(2, fmt="MOVIE", episodes=1, prequel=1, sequel=3, start=(2026, 5, 1), end=(2026, 5, 1)),
        3: media(3, episodes=13, prequel=2, sequel=4, start=(2026, 7, 5), end=(2026, 9, 27)),
        4: media(4, episodes=None, prequel=3, start=(2027, 1, 10), end=None),
    }
    monkeypatch.setattr(tools, "_one", entries.get)
    monkeypatch.setattr(tools.anilist, "with_relations", lambda ids: [entries[i] for i in ids if i in entries])
    return entries


def test_normalize_unifies_script_and_punctuation():
    assert chinese.normalize("小書痴的下剋上：為了") == chinese.normalize("小书痴的下剋上 为了")
    assert chinese.normalize("LV999的村民") == chinese.normalize("LV999 的村民")


def test_chinese_search_prefers_shortest_containing_name(monkeypatch):
    monkeypatch.setattr(chinese, "_index", lambda: [("相反的你和我 第二季", 2), ("相反的你和我", 1)])
    assert chinese.search("相反的你和我") == [1, 2]


def test_chinese_search_falls_back_to_names_inside_query(monkeypatch):
    monkeypatch.setattr(chinese, "_index", lambda: [("相反的你和我", 1), ("你和", 9), ("其他番", 5)])
    # "你和" is shorter than 3 characters, too generic to count as a hit.
    assert chinese.search("完全相反的你和我") == [1]


def test_season_chain_offsets_skip_non_series_formats(franchise):
    chain = tools.season_chain(3)
    assert [c["id"] for c in chain] == [1, 2, 3, 4]
    assert [c["episodes_before"] for c in chain] == [0, None, 12, 25]


def test_season_chain_aired_by_date(franchise, monkeypatch):
    monkeypatch.setattr(tools.anilist, "last_aired", lambda id, before: {3: 1}[id])
    chain = tools.season_chain(1, on_date="2026-07-05T13:00:00Z")
    # S1 finished, S2 airing (schedule lookup), S3 not started.
    assert [c["aired_by_date"] for c in chain] == [12, 1, 1, 0]


def test_get_anime_reports_missing_id(monkeypatch, no_chinese_names):
    monkeypatch.setattr(tools, "_one", lambda id: None)
    assert "error" in tools.get_anime(42)


def test_season_chain_orders_by_air_date_despite_wrong_edges(monkeypatch, no_chinese_names):
    """AniList marks Bananya season 2 (2019) as a PREQUEL of season 1 (2016)."""
    entries = {
        1: media(1, episodes=13, prequel=2, start=(2016, 7, 4), end=(2016, 9, 26)),
        2: media(2, episodes=13, sequel=1, start=(2019, 10, 1), end=(2019, 12, 24)),
    }
    entries[1]["relations"]["edges"].append({"relationType": "SEQUEL", "node": {"id": 3, "type": "ANIME"}})
    entries[3] = media(3, fmt="ONA", episodes=13, prequel=1, start=(2024, 10, 15), end=(2025, 1, 7))
    monkeypatch.setattr(tools.anilist, "with_relations", lambda ids: [entries[i] for i in ids if i in entries])
    chain = tools.season_chain(1)
    assert [c["id"] for c in chain] == [1, 2, 3]
    assert [c["episodes_before"] for c in chain] == [0, 13, 26]


def test_aired_by_compares_in_japan_time(franchise, monkeypatch):
    """15:30 UTC on 4 Jul is 00:30 on 5 Jul in Japan: S2's start day has begun there."""
    monkeypatch.setattr(tools.anilist, "last_aired", lambda id, before: 1)
    chain = tools.season_chain(1, on_date="2026-07-04T15:30:00Z")
    assert chain[2]["aired_by_date"] == 1
    chain = tools.season_chain(1, on_date="2026-07-04T14:30:00Z")
    assert chain[2]["aired_by_date"] == 0
