"""Two-camera App loop on synthetic clips: both calibrations, fused scoring, undo, recording."""

import cv2
import pytest

from dart_scorer import app as app_mod
from dart_scorer import board
from dart_scorer.config import AppConfig, DetectorConfig
from dart_scorer.sources import Recorder, VideoFile
from synth import H, W
from synth3d import CAM1, CAM2, Cam

AIMS = [("T20", 103, 0), ("D16", 166, 234), ("S5", 135, 342)]


def _write(path, cam, aims):
    rec = Recorder(path, 30, (W, H))
    bg = cam.background()
    for _ in range(6):
        rec.write(bg)
    cur = bg.copy()
    for _, r, a in aims:
        cam.add_dart(cur, board.polar_to_canvas(r * board.PX_PER_MM, a))
        for _ in range(15):
            rec.write(cur)
    rec.release()


@pytest.fixture
def rig(tmp_path):
    out = {}
    for n, pos in ((1, CAM1), (2, CAM2)):
        cam = Cam(pos)
        cal = cam.calibrated(seed=n)
        cal.path = tmp_path / f"cal{n}.json"
        cal.save()
        clip = str(tmp_path / f"clip{n}.avi")
        _write(clip, cam, AIMS)
        out[n] = (clip, str(cal.path))
    return out


def _run(rig, tmp_path, monkeypatch, keys=None, **over):
    calls = {"n": 0}
    keys = keys or {}

    def wait_key(_ms):
        calls["n"] += 1
        return keys.get(calls["n"], 255)

    for name in ("namedWindow", "setMouseCallback", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    cfg = AppConfig(
        video=rig[1][0],
        calibration_file=rig[1][1],
        video2=rig[2][0],
        calibration_file2=rig[2][1],
        log_file=str(tmp_path / "t.csv"),
        players=["A"],
        detector=DetectorConfig(settle_frames=5),
        **over,
    )
    a = app_mod.App(cfg)
    a.run()
    return a, cfg


def test_dual_app_scores_with_fusion(rig, tmp_path, monkeypatch):
    a, cfg = _run(rig, tmp_path, monkeypatch, keys={4: ord("b")}, record=str(tmp_path / "rec.avi"))
    assert [s.label for s in a.game.turn_darts] == ["T20", "D16", "S5"]
    rows = open(cfg.log_file).read().splitlines()
    assert len(rows) == 4 and all(r.endswith("auto-intersection") for r in rows[1:])
    r1, r2 = VideoFile(str(tmp_path / "rec.avi")), VideoFile(str(tmp_path / "rec_cam2.avi"))
    assert r1.is_open() and r2.is_open()


def test_dual_app_undo_rearms_both_detectors(rig, tmp_path, monkeypatch):
    # 'u' after the 3rd dart is scored: game rolls back and BOTH detectors leave WAIT_REMOVAL
    a, _ = _run(rig, tmp_path, monkeypatch, keys={4: ord("b"), 49: ord("u")})
    assert len(a.game.turn_darts) == 2
    assert a.det2 is a.ctl.det2
    assert a.det.darts_this_turn == 2 and a.det2.darts_this_turn == 2


def test_single_camera_config_unchanged(rig, tmp_path, monkeypatch):
    cfg = AppConfig(video=rig[1][0], calibration_file=rig[1][1], log_file=str(tmp_path / "s.csv"))
    assert not cfg.dual
    assert app_mod.App(cfg).cam2 is None


def test_dual_replay_cli(rig, tmp_path, capsys):
    from dart_scorer.replay import main

    exp = tmp_path / "e.txt"
    exp.write_text("T20 D16 S5\n")
    code = main(
        [
            rig[1][0],
            "--video2",
            rig[2][0],
            "--calibration",
            rig[1][1],
            "--calibration2",
            rig[2][1],
            "--expect",
            str(exp),
            "--settle-frames",
            "5",
        ]
    )
    assert code == 0
    assert "3/3 correct" in capsys.readouterr().out
