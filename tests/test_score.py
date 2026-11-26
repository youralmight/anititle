import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))

from run import score  # noqa: E402

MATCH = {"status": "matched", "anilist_id": 1, "episode": 2, "error_type": None}
NOT_ANIME = {"status": "error", "anilist_id": None, "episode": None, "error_type": "not_anime"}


def test_score_matched():
    assert score(MATCH, MATCH) == "correct"
    assert score(MATCH, {**MATCH, "episode": 3}) == "wrong_answer"
    assert score(MATCH, NOT_ANIME) == "false_error"


def test_score_expected_error():
    assert score(NOT_ANIME, NOT_ANIME) == "correct"
    assert score(NOT_ANIME, MATCH) == "forced_answer"
    assert score(NOT_ANIME, {**NOT_ANIME, "error_type": "not_found"}) == "correct_wrong_error_type"
