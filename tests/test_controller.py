import csv

import numpy as np
import pytest

from dart_scorer import board
from dart_scorer.calibration import Calibrator
from dart_scorer.config import DetectorConfig, parse_args, parse_bool
from dart_scorer.controller import Action, Controller, ThrowLogger
from dart_scorer.detector import DartDetector, State
from dart_scorer.game import GameState

FRAME = np.full((60, 80), 100, np.uint8)


def flat(sector_mid_angle, r_mm):
    return board.polar_to_canvas(r_mm * board.PX_PER_MM, sector_mid_angle)


@pytest.fixture
def ctl(tmp_path):
    cal = Calibrator(str(tmp_path / "c.json"))
    det = DartDetector(DetectorConfig(shadow_v_only=False))
    det.capture_baseline(FRAME)
    log = ThrowLogger(str(tmp_path / "throws.csv"))
    c = Controller(GameState.new(["A", "B"], 501), det, cal, log)
    c.csv = tmp_path / "throws.csv"
    return c


def rows(c):
    with c.csv.open() as f:
        return list(csv.DictReader(f))


def key(c, text):
    for ch in text:
        c.handle_key(ord(ch), FRAME)


def test_score_updates_game_marks_and_csv(ctl):
    s = ctl.score((10, 20), flat(0, 103))
    assert s.label == "T20" and ctl.game.player.remaining == 441
    assert len(ctl.marks) == 1
    r = rows(ctl)
    assert (
        len(r) == 1
        and r[0]["label"] == "T20"
        and r[0]["source"] == "auto"
        and r[0]["raw_x"] == "10.0"
    )


def test_rejected_darts_not_logged_or_marked(ctl):
    """Regression: darts after the turn is full used to be logged and marked."""
    for _ in range(3):
        ctl.score((0, 0), flat(0, 103))
    assert ctl.score((0, 0), flat(0, 103)) is None
    assert len(rows(ctl)) == 3 and len(ctl.marks) == 3
    assert "Turn over" in ctl.message


def test_darts_after_bust_not_logged(ctl):
    ctl.game.players[0].remaining = ctl.game.turn_start_remaining = 20
    ctl.score((0, 0), flat(0, 103))  # T20 -> bust
    assert ctl.game.busted
    assert ctl.score((0, 0), flat(0, 103)) is None
    assert len(rows(ctl)) == 1


def test_undo_pops_mark_logs_void_and_rearms(ctl):
    for _ in range(3):
        ctl.score((0, 0), flat(0, 103))
    ctl.det.state = State.WAIT_REMOVAL
    ctl.det.darts_this_turn = 3
    ctl.handle_key(ord("u"), FRAME)
    assert len(ctl.marks) == 2 and ctl.game.player.remaining == 501 - 120
    assert ctl.det.state == State.WAIT_THROW and ctl.det.darts_this_turn == 2
    assert rows(ctl)[-1]["source"] == "undo"
    assert ctl.handle_key(ord("u"), FRAME) is Action.CONTINUE


def test_undo_with_nothing_to_undo(ctl):
    assert ctl.undo(FRAME) is False


def test_board_click_overrides_last_dart(ctl):
    ctl.board_click(*flat(0, 103))  # nothing thrown: ignored
    assert rows(ctl) == []
    ctl.score((0, 0), flat(0, 103))
    ctl.board_click(*flat(342, 135))  # S5
    assert ctl.game.turn_darts[-1].label == "S5" and ctl.game.player.remaining == 496
    assert ctl.marks[-1][1].label == "S5" and len(ctl.marks) == 1
    assert rows(ctl)[-1]["source"] == "click_override"


def test_typed_override_replaces_last_dart(ctl):
    ctl.score((0, 0), flat(0, 103))
    key(ctl, "e")
    assert ctl.edit_buffer == ""
    key(ctl, "d16")
    assert ctl.edit_buffer == "D16"
    ctl.handle_key(13, FRAME)
    assert ctl.edit_buffer is None and ctl.game.turn_darts[-1].label == "D16"
    assert rows(ctl)[-1]["source"] == "key_override"


def test_typed_score_with_no_dart_adds_one_with_mark(ctl):
    key(ctl, "eb")  # B = bull
    ctl.handle_key(13, FRAME)
    assert ctl.game.player.remaining == 451 and len(ctl.marks) == 1 and ctl.marks[0][0] is None
    ctl.undo(FRAME)
    assert ctl.marks == [] and ctl.game.player.remaining == 501


def test_typed_invalid_cancel_backspace(ctl):
    key(ctl, "et")
    ctl.handle_key(13, FRAME)
    assert ctl.message == "Invalid entry" and ctl.edit_buffer is None and rows(ctl) == []
    key(ctl, "es5")
    ctl.handle_key(8, FRAME)
    assert ctl.edit_buffer == "S"
    ctl.handle_key(27, FRAME)
    assert ctl.edit_buffer is None
    key(ctl, "e")
    ctl.handle_key(ord("z"), FRAME)  # not an allowed character
    ctl.handle_key(255, FRAME)
    assert ctl.edit_buffer == ""


def test_typed_score_ignored_after_game_won(ctl):
    ctl.game.winner = "A"  # finished game, new empty turn
    key(ctl, "e20")
    ctl.handle_key(13, FRAME)
    assert ctl.marks == [] and rows(ctl) == [] and "Ignored" in ctl.message


def test_next_turn_baseline_quit_recalibrate(ctl):
    ctl.score((0, 0), flat(0, 103))
    ctl.handle_key(ord("n"), FRAME)
    assert ctl.game.player.name == "B" and ctl.marks == [] and "B to throw" in ctl.message
    ctl.det.state = State.WAIT_BASELINE
    ctl.handle_key(ord("b"), FRAME)
    assert ctl.det.state == State.WAIT_THROW
    assert ctl.handle_key(ord("c"), FRAME) is Action.RECALIBRATE
    ctl.after_recalibrate()
    assert ctl.det.state == State.WAIT_BASELINE
    assert ctl.handle_key(ord("q"), FRAME) is Action.QUIT


def test_throw_logger_does_not_rewrite_header(tmp_path):
    p = str(tmp_path / "t.csv")
    ThrowLogger(p)
    ThrowLogger(p)
    assert len(open(p).read().splitlines()) == 1


def test_parse_args_defaults_and_overrides():
    cfg = parse_args([])
    assert cfg.game == 501 and cfg.double_out and cfg.video == "" and cfg.record == ""
    cfg = parse_args(
        [
            "--game",
            "301",
            "--no-double-out",
            "--players",
            "X",
            "Y",
            "Z",
            "--diff-threshold",
            "40",
            "--shadow-v-only",
            "false",
            "--tip-method",
            "lowest",
            "--camera-side",
            "left",
            "--video",
            "a.mp4",
            "--record",
            "b.avi",
        ]
    )
    assert (cfg.game, cfg.double_out, cfg.players) == (301, False, ["X", "Y", "Z"])
    assert cfg.detector.diff_threshold == 40 and cfg.detector.shadow_v_only is False
    assert cfg.detector.tip_method == "lowest" and cfg.detector.camera_side == "left"
    assert cfg.video == "a.mp4" and cfg.record == "b.avi"


def test_parse_bool_explicit_values():
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool("true") is True
    assert parse_bool("yes") is True
    with pytest.raises(Exception):
        parse_bool("maybe")
