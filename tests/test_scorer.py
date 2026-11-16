import math
import pytest
from dart_scorer import board
from dart_scorer.scorer import score_board_mm, parse_score
from dart_scorer.game import GameState


def at(r_mm, angle_deg):
    """Board-mm point at radius r, clockwise-from-top angle."""
    a = math.radians(angle_deg)
    return score_board_mm(r_mm * math.sin(a), -r_mm * math.cos(a))


@pytest.mark.parametrize("r,label,pts", [
    (0.0, "Bullseye", 50), (6.35, "Bullseye", 50), (6.36, "Outer bull", 25),
    (15.9, "Outer bull", 25), (15.95, "S20", 20),
    (99.0, "S20", 20), (99.05, "T20", 60), (107.0, "T20", 60), (107.05, "S20", 20),
    (162.0, "S20", 20), (162.05, "D20", 40), (170.0, "D20", 40), (170.05, "Miss", 0),
])
def test_radial_boundaries(r, label, pts):
    s = at(r, 0)
    assert (s.label, s.points) == (label, pts)


def test_bull_is_double():
    assert at(0, 0).is_double and not at(10, 0).is_double


def test_sector_order_and_wires():
    order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    for i, n in enumerate(order):
        assert at(130, i * 18).sector == n
    # wire at +9 deg belongs to the sector counted from the left of the wire
    assert at(130, 8.99).sector == 20 and at(130, 9.01).sector == 1
    assert at(130, -8.99).sector == 20 and at(130, -9.01).sector == 5


def test_cardinal_calibration_points():
    assert at(170, 0).sector == 20 and at(170, 90).sector == 6
    assert at(170, 180).sector == 3 and at(170, 270).sector == 11


def test_mm_matches_pixel_scale():
    assert board.DOUBLE_OUTER_PX / board.PX_PER_MM == pytest.approx(170.0)


@pytest.mark.parametrize("t,label,pts", [("T20", "T20", 60), ("d16", "D16", 32), ("5", "S5", 5),
                                         ("50", "Bullseye", 50), ("25", "Outer bull", 25), ("0", "Miss", 0)])
def test_parse(t, label, pts):
    s = parse_score(t)
    assert (s.label, s.points) == (label, pts)


def test_game_double_out_and_bust():
    g = GameState.new(["A"], start=40)
    g.add_dart(parse_score("S20"))
    r = g.add_dart(parse_score("D10"))
    assert r.status == "win" and g.winner == "A"
    g = GameState.new(["A"], start=32)
    assert g.add_dart(parse_score("S16")).status == "ok"
    assert g.add_dart(parse_score("S16")).status == "bust"  # 0 on a single
    assert g.player.remaining == 32
    g = GameState.new(["A"], start=50)
    assert g.add_dart(parse_score("50")).status == "win"    # bull checks out
