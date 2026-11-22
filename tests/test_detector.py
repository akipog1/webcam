import cv2
import numpy as np

from dart_scorer.config import DetectorConfig
from dart_scorer.detector import DartDetector, State, estimate_tip

H, W = 240, 320


def blank():
    return np.full((H, W), 100, np.uint8)


def with_darts(darts):
    f = blank()
    for x1, y1, x2, y2 in darts:
        cv2.line(f, (x1, y1), (x2, y2), 220, 4)
    return f


def cfg():
    return DetectorConfig(
        settle_frames=3,
        min_area=20,
        max_area=3000,
        hand_motion_area=8000,
        shadow_v_only=False,
        tip_method="lowest",
    )


def feed(d, frame, n):
    out = []
    for _ in range(n):
        r = d.update(frame)
        if r:
            out.append(r)
        d.after_scored()
    return out


def test_waits_for_baseline():
    d = DartDetector(cfg())
    assert d.update(with_darts([(50, 50, 80, 120)])) is None
    assert d.state == State.WAIT_BASELINE


def test_identical_frames_skip_processing(monkeypatch):
    d = DartDetector(cfg())
    frame = blank()
    d.capture_baseline(frame)
    calls = 0
    original = d.preprocess

    def counted_preprocess(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(d, "preprocess", counted_preprocess)
    assert d.update(frame) is None
    assert calls == 0


def test_scores_each_dart_once_and_only_newest():
    d = DartDetector(cfg())
    d.capture_baseline(blank())
    a = (50, 40, 70, 120)
    b = (200, 40, 220, 150)
    hits = feed(d, with_darts([a]), 20)
    assert len(hits) == 1
    assert abs(hits[0].tip[1] - 120) < 4  # lowest point = tip
    hits2 = feed(d, with_darts([a, b]), 20)
    assert len(hits2) == 1
    assert abs(hits2[0].tip[0] - 220) < 5


def test_requires_settling():
    d = DartDetector(cfg())
    d.capture_baseline(blank())
    assert feed(d, with_darts([(50, 40, 70, 120)]), 2) == []


def test_ignores_hand_sized_motion():
    d = DartDetector(cfg())
    d.capture_baseline(blank())
    hand = blank()
    hand[20:200, 20:300] = 250
    assert feed(d, hand, 20) == []


def test_three_darts_then_wait_removal_then_reset():
    d = DartDetector(cfg())
    d.capture_baseline(blank())
    darts = [(40, 40, 60, 120), (140, 40, 160, 120), (240, 40, 260, 120)]
    total = 0
    for i in range(3):
        total += len(feed(d, with_darts(darts[: i + 1]), 20))
    assert total == 3 and d.state == State.WAIT_REMOVAL
    feed(d, blank(), 10)
    assert d.state == State.WAIT_THROW and d.darts_this_turn == 0


def test_round_blob_rejected_by_aspect():
    d = DartDetector(cfg())
    d.capture_baseline(blank())
    f = blank()
    cv2.circle(f, (100, 100), 12, 220, -1)
    assert feed(d, f, 20) == []


def test_fitline_tip_toward_camera_side():
    f = np.zeros((200, 200), np.uint8)
    cv2.line(f, (60, 30), (120, 170), 255, 5)
    c, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y = estimate_tip(c[0], "fitline", "bottom")
    assert np.hypot(x - 120, y - 170) < 1.5


def test_fitline_tip_rejects_an_isolated_extreme_mask_pixel():
    f = np.zeros((200, 200), np.uint8)
    cv2.line(f, (60, 30), (120, 170), 255, 5)
    c, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    noisy_contour = np.concatenate((c[0], np.array([[[124, 179]]], dtype=np.int32)))

    x, y = estimate_tip(noisy_contour, "fitline", "bottom")

    assert np.hypot(x - 120, y - 170) < 2


def test_fitline_tip_resists_an_off_axis_contour_lobe():
    axis = np.array([0.4472136, 0.8944272], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    center = np.array([80, 90], dtype=np.float32)
    shaft = np.array(
        [
            center + along * axis + across * normal
            for along in np.linspace(-60, 60, 121)
            for across in (-2, 2)
        ],
        dtype=np.float32,
    )
    lobe = np.array(
        [
            center + along * axis + (35 + across) * normal
            for along in np.linspace(-50, -30, 20)
            for across in (-2, 2)
        ],
        dtype=np.float32,
    )
    contour = np.vstack((shaft, lobe)).reshape(-1, 1, 2)
    expected_tip = center + 60 * axis

    x, y = estimate_tip(contour, "fitline", "bottom")

    assert np.hypot(x - expected_tip[0], y - expected_tip[1]) < 3
