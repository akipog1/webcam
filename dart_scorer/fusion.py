"""Two-camera fusion.

Each camera's homography maps image pixels to the board plane. The dart tip is
*on* that plane, so its pixel maps to the right board position, but the rest of
the dart (barrel, flight) sticks out of the plane and maps to the wrong place.
Mapping the dart's whole axis line from camera A onto the board gives a line
that still passes through the true tip; camera B gives another such line. Their
intersection is the tip, independent of the depth ambiguity a single camera has.

Fallbacks (``FusedTip.method``): ``intersection`` -> ``average`` of both
single-camera tips (lines unusable / nearly parallel) -> ``cam1`` / ``cam2``
(only one camera saw the dart).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import board
from .calibration import Calibrator
from .detector import Detection

Point = tuple[float, float]
# a line in the board plane: (point, unit direction)
Line = tuple[np.ndarray, np.ndarray]

MIN_ELONGATION = 2.5          # sqrt(major/minor variance) needed to trust a contour's axis
MAX_RADIUS_PX = board.DOUBLE_OUTER_PX * 1.15   # intersections farther out are rejected


@dataclass
class FusedTip:
    flat: Point            # board-canvas coordinates
    method: str            # intersection | average | cam1 | cam2
    raw: Point             # camera pixel of the (first available) single-camera tip, for logging


def board_line(det: Detection, cal: Calibrator, span: float = 60.0) -> Optional[Line]:
    """The dart's axis, mapped onto the board plane; None if the contour is not elongated enough."""
    pts = det.contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return None
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T)
    evals, evecs = np.linalg.eigh(cov)           # ascending
    if evals[0] <= 1e-9 or math.sqrt(evals[1] / evals[0]) < MIN_ELONGATION:
        return None
    d = evecs[:, 1]
    a = cal.to_board((centre[0] - d[0] * span, centre[1] - d[1] * span))
    b = cal.to_board((centre[0] + d[0] * span, centre[1] + d[1] * span))
    v = np.array(b) - np.array(a)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return np.array(a), v / n


def intersect(l1: Line, l2: Line, min_angle_deg: float) -> Optional[Point]:
    """Intersection of two board-plane lines, or None if they are too close to parallel."""
    (p1, d1), (p2, d2) = l1, l2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < math.sin(math.radians(min_angle_deg)):
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    q = p1 + t * d1
    return float(q[0]), float(q[1])


def fuse(d1: Optional[Detection], cal1: Calibrator,
         d2: Optional[Detection], cal2: Calibrator,
         min_angle_deg: float = 20.0) -> Optional[FusedTip]:
    """Combine up to two detections of the same dart into one board position."""
    if d1 is None and d2 is None:
        return None
    if d2 is None:
        return FusedTip(cal1.to_board(d1.tip), "cam1", d1.tip)
    if d1 is None:
        return FusedTip(cal2.to_board(d2.tip), "cam2", d2.tip)
    l1, l2 = board_line(d1, cal1), board_line(d2, cal2)
    if l1 is not None and l2 is not None:
        p = intersect(l1, l2, min_angle_deg)
        if p is not None and math.hypot(p[0] - board.CENTER[0], p[1] - board.CENTER[1]) <= MAX_RADIUS_PX:
            return FusedTip(p, "intersection", d1.tip)
    a, b = cal1.to_board(d1.tip), cal2.to_board(d2.tip)
    return FusedTip(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), "average", d1.tip)


class DetectionPairer:
    """Matches detections of the same dart from two independent detectors.

    Each detector settles on its own schedule. A detection waits up to
    ``window`` frames for the other camera; if none arrives it is emitted alone.
    """

    def __init__(self, window: int = 45) -> None:
        self.window = window
        self._pending: dict[int, tuple[int, Detection]] = {}
        self._frame = 0

    def push(self, cam: int, det: Detection) -> list[tuple[Optional[Detection], Optional[Detection]]]:
        """Register a detection from camera 1 or 2; returns pairs that are ready."""
        other = 2 if cam == 1 else 1
        if other in self._pending:
            _, od = self._pending.pop(other)
            return [(det, od) if cam == 1 else (od, det)]
        if cam in self._pending:                    # same camera twice: flush the old one alone
            _, old = self._pending.pop(cam)
            out = [(old, None) if cam == 1 else (None, old)]
        else:
            out = []
        self._pending[cam] = (self._frame, det)
        return out

    def tick(self) -> list[tuple[Optional[Detection], Optional[Detection]]]:
        """Call once per frame; flushes detections whose partner never came."""
        self._frame += 1
        out = []
        for cam, (f, det) in list(self._pending.items()):
            if self._frame - f >= self.window:
                del self._pending[cam]
                out.append((det, None) if cam == 1 else (None, det))
        return out

    def clear(self) -> None:
        self._pending.clear()
