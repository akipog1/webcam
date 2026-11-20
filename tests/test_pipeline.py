"""End-to-end test on synthetic images: render a board and darts through a
perspective camera, calibrate from simulated clicks, then run detector ->
homography -> scorer -> game exactly as the app does (minus the GUI)."""
import json

import numpy as np
import pytest

from dart_scorer import board
from dart_scorer.calibration import Calibrator
from dart_scorer.detector import DartDetector, State
from dart_scorer.game import GameState
from dart_scorer.scorer import score_point
from synth import (AIMS, H, W, add_dart, calibrated, cfg, ground_truth, render_board,
                   run, to_cam)


@pytest.fixture(scope="module")
def world():
    return ground_truth()


def test_calibration_recovers_geometry(world):
    _, h_canvas2cam = world
    cal, err = calibrated(h_canvas2cam)
    assert err < 3.0
    # A known canvas point maps back to within a couple of canvas px.
    pt = board.polar_to_canvas(250, 77)
    back = cal.to_board(to_cam(h_canvas2cam, pt))
    assert np.hypot(back[0] - pt[0], back[1] - pt[1]) < 4


def test_full_turn_scores_and_game(world):
    _, h_canvas2cam = world
    cal, _ = calibrated(h_canvas2cam)
    empty = render_board(h_canvas2cam)
    det = DartDetector(cfg())
    det.capture_baseline(empty)
    game = GameState.new(["A", "B"], 501)

    frame = empty.copy()
    for label, aim in AIMS:
        add_dart(frame, to_cam(h_canvas2cam, aim))
        hits = run(det, frame)
        assert len(hits) == 1, f"{label}: expected exactly one detection"
        flat = cal.to_board(hits[0].tip)
        s = score_point(*flat)
        assert s.label == label, f"{label}: scored {s.label} at {flat}"
        game.add_dart(s)

    assert det.state == State.WAIT_REMOVAL
    assert game.player.remaining == 501 - (60 + 32 + 5)
    assert game.turn_over

    # Darts pulled -> detector re-arms by itself.
    run(det, empty, 12)
    assert det.state == State.WAIT_THROW and det.darts_this_turn == 0


@pytest.mark.parametrize("label,polar", [
    ("Bullseye", (0.0, 0)),
    ("Outer bull", (11 * board.PX_PER_MM, 90)),
])
def test_bull_regions(world, label, polar):
    _, h_canvas2cam = world
    cal, _ = calibrated(h_canvas2cam)
    empty = render_board(h_canvas2cam)
    det = DartDetector(cfg())
    det.capture_baseline(empty)
    frame = empty.copy()
    add_dart(frame, to_cam(h_canvas2cam, board.polar_to_canvas(*polar)))
    hits = run(det, frame)
    assert len(hits) == 1
    assert score_point(*cal.to_board(hits[0].tip)).label == label


def test_miss_off_board(world):
    _, h_canvas2cam = world
    cal, _ = calibrated(h_canvas2cam)
    empty = render_board(h_canvas2cam)
    det = DartDetector(cfg())
    det.capture_baseline(empty)
    frame = empty.copy()
    add_dart(frame, to_cam(h_canvas2cam, board.polar_to_canvas(215 * board.PX_PER_MM, 40)))
    hits = run(det, frame)
    assert len(hits) == 1
    assert score_point(*cal.to_board(hits[0].tip)).label == "Miss"


def test_hand_in_frame_scores_nothing(world):
    _, h_canvas2cam = world
    empty = render_board(h_canvas2cam)
    det = DartDetector(cfg())
    det.capture_baseline(empty)
    frame = empty.copy()
    frame[100:500, 300:800] = (150, 170, 220)
    assert run(det, frame, 30) == []


def test_undo_after_third_dart_rearms_detector(world):
    """Regression: undo used to leave the detector stuck in WAIT_REMOVAL."""
    _, h_canvas2cam = world
    empty = render_board(h_canvas2cam)
    det = DartDetector(cfg())
    det.capture_baseline(empty)
    frame = empty.copy()
    for _, aim in AIMS:
        add_dart(frame, to_cam(h_canvas2cam, aim))
        run(det, frame)
    assert det.state == State.WAIT_REMOVAL
    det.undo_dart(frame)
    assert det.state == State.WAIT_THROW and det.darts_this_turn == 2
    # The (still present) three darts must not be re-scored...
    assert run(det, frame, 20) == []
    # ...but a new dart is.
    add_dart(frame, to_cam(h_canvas2cam, board.polar_to_canvas(60, 180)))
    assert len(run(det, frame)) == 1


def test_calibration_file_roundtrip_and_size_check(world, tmp_path):
    _, h_canvas2cam = world
    cal, _ = calibrated(h_canvas2cam)
    cal.path = tmp_path / "cal.json"
    cal.frame_size = (W, H)
    cal.save()
    ok = Calibrator(str(cal.path))
    assert ok.load((W, H))
    assert np.allclose(ok.H, cal.H)
    wrong = Calibrator(str(cal.path))
    assert not wrong.load((640, 480))
    assert wrong.H is None


def test_calibration_rejects_malformed_homography(tmp_path):
    cal = Calibrator(str(tmp_path / "cal.json"))
    for bad in (
        np.zeros((2, 3)),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        np.array([[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]], dtype=float),
    ):
        with pytest.raises(ValueError):
            cal.set_matrix(bad)

    cal.path.write_text(json.dumps({
        "canvas": board.CANVAS_SIZE,
        "frame": [W, H],
        "H": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    }))
    assert not cal.load((W, H))
    assert cal.H is None
