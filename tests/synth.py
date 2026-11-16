"""Synthetic scene helpers: a perspective camera looking at a rendered board."""
import cv2
import numpy as np

from dart_scorer import board
from dart_scorer.calibration import Calibrator
from dart_scorer.config import DetectorConfig

W, H = 1280, 720
# Camera positions of the four outer-double-wire points (top, right, bottom, left):
# a mildly rotated, perspective-squashed view of the board.
CAM_WIRE = np.array([[655, 150], [940, 372], [628, 600], [352, 388]], np.float32)
CANVAS_WIRE = np.array(list(board.calibration_targets().values())[1:], np.float32)


def ground_truth():
    """Ground-truth (camera->canvas, canvas->camera) homographies."""
    h_cam2canvas, _ = cv2.findHomography(CAM_WIRE, CANVAS_WIRE)
    return h_cam2canvas, np.linalg.inv(h_cam2canvas)


def to_cam(h_canvas2cam, pt):
    p = cv2.perspectiveTransform(np.array([[pt]], np.float32), h_canvas2cam)[0, 0]
    return float(p[0]), float(p[1])


def render_board(h_canvas2cam):
    """Textured, boardlike background so the baseline is not trivially flat."""
    canvas = np.full((board.CANVAS_SIZE,) * 2 + (3,), 60, np.uint8)
    c = (int(board.CENTER[0]), int(board.CENTER[1]))
    cv2.circle(canvas, c, int(board.DOUBLE_OUTER_PX) + 40, (30, 30, 30), -1)
    for i in range(20):
        a0, a1 = i * 18 - 9 - 90, i * 18 + 9 - 90
        col = (200, 200, 200) if i % 2 else (40, 40, 40)
        cv2.ellipse(canvas, c, (int(board.DOUBLE_OUTER_PX),) * 2, 0, a0, a1, col, -1)
    for r in board.RING_RADII_PX:
        cv2.circle(canvas, c, int(round(r)), (0, 0, 200), 2)
    return cv2.warpPerspective(canvas, h_canvas2cam, (W, H), borderValue=(70, 70, 70))


def add_dart(frame, tip):
    """Dart leaning away from a camera mounted below: tip at the bottom end."""
    x, y = int(round(tip[0])), int(round(tip[1]))
    cv2.line(frame, (x, y), (x + 8, y - 70), (255, 255, 255), 5, cv2.LINE_AA)


def cfg():
    return DetectorConfig(settle_frames=5, min_area=40)


def run(det, frame, n=15):
    hits = []
    for _ in range(n):
        d = det.update(frame)
        det.after_scored()
        if d:
            hits.append(d)
    return hits


def calibrated(h_canvas2cam, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    cal = Calibrator()
    names = list(cal.targets)
    clicks = [to_cam(h_canvas2cam, cal.targets[n]) for n in names]
    clicks = [tuple(np.array(c) + rng.uniform(-noise, noise, 2)) for c in clicks]
    err = cal.fit(clicks, [cal.targets[n] for n in names])
    return cal, err




# Aim points in the middle of each region (away from wires) -> expected label.
AIMS = [
    ("T20", board.polar_to_canvas(103 * board.PX_PER_MM, 0)),
    ("D16", board.polar_to_canvas(166 * board.PX_PER_MM, 234)),   # 16 spans 234 deg
    ("S5", board.polar_to_canvas(135 * board.PX_PER_MM, 342)),    # 5 spans 342 deg
]
