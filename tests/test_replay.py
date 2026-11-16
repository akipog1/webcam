import cv2
import numpy as np
import pytest

from dart_scorer import board
from dart_scorer.config import DetectorConfig
from dart_scorer.replay import format_table, main, parse_expected, replay
from dart_scorer.sources import Recorder, VideoFile
from synth import AIMS, H, W, add_dart, calibrated, cfg, ground_truth, render_board, to_cam


@pytest.fixture(scope="module")
def scene():
    _, h = ground_truth()
    cal, _ = calibrated(h)
    empty = render_board(h)
    frames = [empty.copy() for _ in range(8)]       # empty board; baseline at frame 5
    cur = empty.copy()
    for _, aim in AIMS:
        add_dart(cur, to_cam(h, aim))
        frames += [cur.copy() for _ in range(15)]
    frames += [empty.copy() for _ in range(12)]     # darts pulled
    cur = empty.copy()
    add_dart(cur, to_cam(h, board.polar_to_canvas(30 * board.PX_PER_MM, 200)))
    frames += [cur.copy() for _ in range(15)]       # first dart of next turn
    return cal, frames


def test_parse_expected():
    got = parse_expected("T20, D16 S5  # comment\n50\n5\n")
    assert [s.label for s in got] == ["T20", "D16", "S5", "Bullseye", "S5"]
    with pytest.raises(ValueError):
        parse_expected("T99")


def test_replay_matches_expected(scene):
    cal, frames = scene
    res = replay(frames, cal, cfg(), parse_expected("T20 D16 S5 S19"), baseline_frame=5)
    assert res.passed and res.accuracy == 1.0 and len(res.rows) == 4
    assert res.rows[0].frame > 5 and res.frames == len(frames)
    assert "4/4 correct" in format_table(res)


def test_replay_flags_wrong_miss_and_extra(scene):
    cal, frames = scene
    res = replay(frames, cal, cfg(), parse_expected("T20 D1 S5"), baseline_frame=5)
    verdicts = [(r.expected.label if r.expected else None, r.ok) for r in res.rows]
    assert verdicts == [("T20", True), ("D1", False), ("S5", True), (None, False)]
    table = format_table(res)
    assert "WRONG" in table and "EXTRA" in table and not res.passed
    assert res.accuracy == pytest.approx(2 / 3)

    res = replay(frames[:30], cal, cfg(), parse_expected("T20 D16 S5"), baseline_frame=5)
    assert [r.ok for r in res.rows][-1] is False and "MISS" in format_table(res)


def test_recorder_and_videofile_roundtrip(tmp_path, scene):
    _, frames = scene
    for name in ("clip.avi", "clip.mp4"):
        path = str(tmp_path / name)
        rec = Recorder(path, 30, (W, H))
        for f in frames[:10]:
            rec.write(f)
        rec.write(cv2.resize(frames[0], (320, 240)))   # wrong size is resized, not dropped
        rec.release()
        vf = VideoFile(path)
        assert vf.is_open() and vf.size == (W, H) and not vf.is_live
        n = 0
        while vf.read() is not None:
            n += 1
        assert n == 11
        vf.rewind()
        assert vf.read() is not None
        vf.release()


def test_recorder_bad_path(tmp_path):
    with pytest.raises(OSError):
        Recorder(str(tmp_path / "nope" / "x.mp4"), 30, (W, H))


def test_cli_end_to_end(tmp_path, scene, capsys):
    cal, frames = scene
    clip = str(tmp_path / "clip.avi")
    rec = Recorder(clip, 30, (W, H))
    for f in frames:
        rec.write(f)
    rec.release()
    cal.path = tmp_path / "cal.json"
    cal.frame_size = (W, H)
    cal.save()
    exp = tmp_path / "exp.txt"
    exp.write_text("T20 D16 S5 S19\n")
    common = [clip, "--calibration", str(cal.path), "--settle-frames", "5", "--min-area", "40"]
    assert main(common + ["--expect", str(exp)]) == 0
    assert "4/4 correct" in capsys.readouterr().out
    exp.write_text("T20 D16 S5 T1\n")
    assert main(common + ["--expect", str(exp)]) == 1
    assert main(common + ["--expect", str(exp), "--min-accuracy", "0.75"]) == 0
    assert main(common) == 0                                   # no --expect: just list detections
    assert main([clip, "--calibration", str(tmp_path / "missing.json")]) == 2
    assert main([str(tmp_path / "none.avi"), "--calibration", str(cal.path)]) == 2
    cal.frame_size = (640, 480); cal.save()                    # wrong size calibration
    assert main(common) == 2
