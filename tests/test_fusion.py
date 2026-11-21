import math

import cv2
import numpy as np
import pytest

from dart_scorer import board
from dart_scorer.config import DetectorConfig
from dart_scorer.detector import DartDetector, Detection
from dart_scorer.fusion import DetectionPairer, fuse
from dart_scorer.scorer import score_point
from synth3d import CAM1, CAM2, Cam

AIMS = [
    ("T20", 103, 0),
    ("D16", 166, 234),
    ("S5", 135, 342),
    ("D6", 166, 90),
    ("T19", 103, 198),
    ("S1", 135, 18),
    ("50", 0, 0),
]


def _detect(cam, cal, tip):
    bg = cam.background()
    fr = bg.copy()
    cam.add_dart(fr, tip)
    det = DartDetector(DetectorConfig(settle_frames=5, min_area=40))
    det.capture_baseline(bg)
    for _ in range(12):
        d = det.update(fr)
        det.after_scored()
        if d:
            return d
    pytest.fail("dart not detected")


@pytest.fixture(scope="module")
def rig():
    c1, c2 = Cam(CAM1), Cam(CAM2)
    return c1, c2, c1.calibrated(seed=1), c2.calibrated(seed=2)


@pytest.mark.parametrize("label,r_mm,ang", AIMS)
def test_intersection_recovers_true_tip(rig, label, r_mm, ang):
    c1, c2, cal1, cal2 = rig
    truth = board.polar_to_canvas(r_mm * board.PX_PER_MM, ang)
    d1, d2 = _detect(c1, cal1, truth), _detect(c2, cal2, truth)
    f = fuse(d1, cal1, d2, cal2)
    assert f.method == "intersection"
    err_mm = math.hypot(f.flat[0] - truth[0], f.flat[1] - truth[1]) / board.PX_PER_MM
    assert err_mm < 4.0
    assert score_point(*f.flat).label.replace("Bullseye", "50") in (label, "Bullseye")


def test_single_camera_cases(rig):
    c1, c2, cal1, cal2 = rig
    truth = board.polar_to_canvas(103 * board.PX_PER_MM, 0)
    d1, d2 = _detect(c1, cal1, truth), _detect(c2, cal2, truth)
    assert fuse(d1, cal1, None, cal2).method == "cam1"
    assert fuse(None, cal1, d2, cal2).method == "cam2"
    assert fuse(None, cal1, None, cal2) is None


def test_parallel_lines_fall_back_to_average(rig):
    c1, _, cal1, _ = rig
    truth = board.polar_to_canvas(103 * board.PX_PER_MM, 0)
    d1 = _detect(c1, cal1, truth)
    f = fuse(d1, cal1, d1, cal1)  # identical view -> lines parallel
    assert f.method == "average"
    assert f.flat == pytest.approx(cal1.to_board(d1.tip))


def test_round_blob_has_no_axis_so_averages(rig):
    c1, c2, cal1, cal2 = rig
    truth = board.polar_to_canvas(103 * board.PX_PER_MM, 0)
    d1, d2 = _detect(c1, cal1, truth), _detect(c2, cal2, truth)
    blob = np.array([[[x, y]] for x, y in [(10, 10), (20, 10), (20, 20), (10, 20)]], np.int32)
    round_det = Detection(d2.tip, d2.centroid, blob, 100.0)
    assert fuse(d1, cal1, round_det, cal2).method == "average"


def _fake(n):
    return Detection((n, n), (n, n), np.zeros((1, 1, 2), np.int32), 1.0)


def test_pairer_pairs_and_times_out():
    p = DetectionPairer(window=3)
    a, b = _fake(1), _fake(2)
    assert p.push(1, a) == []
    assert p.push(2, b) == [(a, b)]
    c = _fake(3)
    p.push(2, c)
    out = []
    for _ in range(3):
        out += p.tick()
    assert out == [(None, c)]
    assert p.tick() == []


def test_pairer_flushes_same_camera_twice():
    p = DetectionPairer(window=50)
    a, b = _fake(1), _fake(2)
    p.push(1, a)
    assert p.push(1, b) == [(a, None)]
