"""Drive the real App loop headlessly: video in, scripted key presses, GUI calls stubbed."""

import cv2
import pytest

from dart_scorer import app as app_mod
from dart_scorer.config import AppConfig, DetectorConfig
from dart_scorer.sources import Recorder, VideoFile
from synth import AIMS, H, W, add_dart, calibrated, ground_truth, render_board, to_cam


@pytest.fixture
def clip(tmp_path):
    _, h = ground_truth()
    cal, _ = calibrated(h)
    cal.path = tmp_path / "cal.json"
    cal.frame_size = (W, H)
    cal.save()
    empty = render_board(h)
    path = str(tmp_path / "clip.avi")
    rec = Recorder(path, 30, (W, H))
    for _ in range(6):
        rec.write(empty)
    cur = empty.copy()
    for _, aim in AIMS:
        add_dart(cur, to_cam(h, aim))
        for _ in range(15):
            rec.write(cur)
    rec.release()
    return path, str(cal.path)


def test_app_replays_clip_and_scores(clip, tmp_path, monkeypatch):
    path, calfile = clip
    calls = {"n": 0}

    def wait_key(_ms):
        calls["n"] += 1
        return ord("b") if calls["n"] == 4 else 255  # baseline while the board is empty

    for name in ("namedWindow", "setMouseCallback", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)

    cfg = AppConfig(
        video=path,
        calibration_file=calfile,
        log_file=str(tmp_path / "t.csv"),
        record=str(tmp_path / "rec.avi"),
        players=["A"],
        detector=DetectorConfig(settle_frames=5),
    )
    a = app_mod.App(cfg)
    a.run()

    assert [s.label for s in a.game.turn_darts] == ["T20", "D16", "S5"]
    assert a.game.player.remaining == 501 - 97
    lines = open(cfg.log_file).read().splitlines()
    assert len(lines) == 4  # header + 3 throws
    rec = VideoFile(cfg.record)  # session was recorded for replay
    assert rec.is_open() and rec.size == (W, H)
