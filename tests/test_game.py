from dart_scorer.game import GameState
from dart_scorer.scorer import parse_score as p


def test_countdown_and_turn_change():
    g = GameState.new(["A", "B"], 501)
    for t in ("T20", "T20", "T20"):
        g.add_dart(p(t))
    assert g.player.remaining == 321
    g.next_turn()
    assert g.player.name == "B" and g.player.remaining == 501
    assert g.players[0].three_dart_average == 180


def test_bust_over():
    g = GameState.new(["A"], 301)
    g.players[0].remaining = g.turn_start_remaining = 40
    g.add_dart(p("T20"))
    assert g.busted and g.player.remaining == 40


def test_double_out_required():
    g = GameState.new(["A"], 301)
    g.players[0].remaining = g.turn_start_remaining = 20
    assert g.add_dart(p("S20")).status == "bust"
    g.undo()
    assert g.add_dart(p("D10")).status == "win"


def test_leaving_one_is_bust_with_double_out():
    g = GameState.new(["A"], 301)
    g.players[0].remaining = g.turn_start_remaining = 21
    assert g.add_dart(p("S20")).status == "bust"


def test_no_double_out():
    g = GameState.new(["A"], 301, double_out=False)
    g.players[0].remaining = g.turn_start_remaining = 20
    assert g.add_dart(p("S20")).status == "win"


def test_bull_counts_as_double():
    g = GameState.new(["A"], 301)
    g.players[0].remaining = g.turn_start_remaining = 50
    assert g.add_dart(p("50")).status == "win"


def test_override_and_undo():
    g = GameState.new(["A"], 501)
    g.add_dart(p("S1"))
    g.override_last(p("T20"))
    assert g.player.remaining == 441
    g.undo()
    assert g.player.remaining == 501


def test_max_three_darts():
    g = GameState.new(["A"], 501)
    for _ in range(3):
        g.add_dart(p("S1"))
    assert g.add_dart(p("S1")).status == "turn_full"
