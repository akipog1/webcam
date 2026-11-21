import math

import pytest

from dart_scorer import board
from dart_scorer.scorer import parse_score, score_point, score_polar


def pt(r, a):
    return board.polar_to_canvas(r, a)


@pytest.mark.parametrize("i,sector", list(enumerate(board.SECTORS)))
def test_every_sector_centre_and_edges(i, sector):
    r = (board.TREBLE_OUTER_PX + board.DOUBLE_INNER_PX) / 2
    for a in (i * 18.0, i * 18.0 - 8.9, i * 18.0 + 8.9):
        s = score_point(*pt(r, a % 360))
        assert s.sector == sector and s.points == sector


def test_twenty_is_top_and_six_is_right():
    assert score_point(500, 300).sector == 20
    assert score_point(700, 500).sector == 6
    assert score_point(500, 700).sector == 3
    assert score_point(300, 500).sector == 11


@pytest.mark.parametrize(
    "radius,ring,mult",
    [
        (0, "inner_bull", 2),
        (board.INNER_BULL_PX, "inner_bull", 2),
        (board.INNER_BULL_PX + 0.01, "outer_bull", 1),
        (board.OUTER_BULL_PX, "outer_bull", 1),
        (board.OUTER_BULL_PX + 0.01, "inner_single", 1),
        (board.TREBLE_INNER_PX, "inner_single", 1),
        (board.TREBLE_INNER_PX + 0.01, "treble", 3),
        (board.TREBLE_OUTER_PX, "treble", 3),
        (board.TREBLE_OUTER_PX + 0.01, "outer_single", 1),
        (board.DOUBLE_INNER_PX, "outer_single", 1),
        (board.DOUBLE_INNER_PX + 0.01, "double", 2),
        (board.DOUBLE_OUTER_PX, "double", 2),
        (board.DOUBLE_OUTER_PX + 0.01, "miss", 0),
        (499, "miss", 0),
    ],
)
def test_ring_boundaries_contiguous(radius, ring, mult):
    s = score_polar(radius, 0)
    assert s.ring == ring and s.multiplier == mult


def test_points():
    assert score_polar(0, 0).points == 50
    assert score_polar(board.OUTER_BULL_PX, 0).points == 25
    assert score_polar(board.TREBLE_INNER_PX + 1, 0).points == 60
    assert score_polar(board.DOUBLE_OUTER_PX - 1, 0).points == 40
    assert score_polar(board.DOUBLE_OUTER_PX + 1, 0).points == 0


def test_radii_from_real_dimensions():
    assert math.isclose(board.TREBLE_OUTER_PX / board.DOUBLE_OUTER_PX, 107 / 170)
    assert math.isclose(board.INNER_BULL_PX / board.DOUBLE_OUTER_PX, 6.35 / 170)


def test_calibration_targets_match_scoring_radius():
    for name, (x, y) in board.calibration_targets().items():
        if name != "bull":
            assert math.isclose(math.hypot(x - 500, y - 500), board.DOUBLE_OUTER_PX)
            assert score_point(x, y).ring == "double"


@pytest.mark.parametrize(
    "txt,pts", [("T20", 60), ("d16", 32), ("S5", 5), ("7", 7), ("25", 25), ("50", 50), ("0", 0)]
)
def test_parse_score(txt, pts):
    assert parse_score(txt).points == pts


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_score("T21")


def test_parse_score_bull_shorthand_and_errors():
    from dart_scorer.scorer import parse_score
    import pytest

    assert parse_score("B").points == 50
    assert parse_score("bull").points == 50
    for bad in ("", "T", "X5", "T21", "S", "BB"):
        with pytest.raises(ValueError):
            parse_score(bad)
