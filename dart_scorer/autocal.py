"""Automatic board detection: finds the dartboard in a frame and returns the
homography (camera px -> flat board canvas) that the manual calibration would
have produced.

Method (no learned models, only OpenCV/numpy):
  1. Bull: the 20 spokes are straight lines that all pass through the bull
     (perspective preserves that), so the point where most line segments
     intersect is the bull. Or pass ``bull=(x, y)`` if the user clicked it.
  2. Pose search around that centre: board size, tilt and rotation. For each
     rotation the 20 single-segment wedges should alternate black/cream; peaks
     repeat every 36 degrees, so the one that puts the 20 closest to image-up
     wins (the same "20 at the top" assumption the manual calibration uses).
  3. Refinement: the ideal wire layout (20 spokes + 6 rings) is fitted to the
     image's edges by chamfer matching, moving the four outer-wire anchor
     points (a full 8-parameter homography) with a coarse-to-fine pattern search.

The result is a proposal: the caller shows it over the live view and the user
accepts or falls back to clicking. ``confidence`` is the fraction of wire
points that land on an image edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import board

WORK_WIDTH = 640  # detection runs on a downscaled copy
R_OUT_MM = 225.0  # outer edge of the black number ring (approx.)
MAX_UP_DEVIATION_DEG = 22.0  # how far the 20 may be rotated away from image-up
ORIENT_STEP_DEG = 3
SCALE_TRIALS = (0.92, 0.96, 1.0, 1.04, 1.08)
SEARCH_STEPS = (10.0, 5.0, 2.5, 1.2, 0.6, 0.3)  # percent of the board radius

_CANVAS_ANCHORS = np.array(
    [board.polar_to_canvas(board.DOUBLE_OUTER_PX, a) for a in (0, 90, 180, 270)], np.float32
)


@dataclass
class AutoResult:
    H: np.ndarray  # camera px (original frame) -> canvas
    confidence: float  # 0..1, fraction of wire points sitting on an edge
    mean_edge_px: float  # mean distance (original px) from wire points to the nearest edge


# ---------- template ----------
def _template_points() -> tuple[np.ndarray, np.ndarray]:
    """Wire sample points in canvas coords plus the unit direction of the wire at each."""
    pts, dirs = [], []
    for i in range(20):
        a = board.sector_wire_angle(i)
        rad = math.radians(a)
        d = (math.sin(rad), -math.cos(rad))  # radial direction (canvas, y down)
        for r in np.arange(board.OUTER_BULL_PX + 6, board.DOUBLE_OUTER_PX - 2, 7.0):
            pts.append(board.polar_to_canvas(float(r), a))
            dirs.append(d)
    for r in board.RING_RADII_PX[1:]:  # inner bull ring is too small to be reliable
        for a in range(0, 360, 2):
            rad = math.radians(a)
            pts.append(board.polar_to_canvas(float(r), float(a)))
            dirs.append((math.cos(rad), math.sin(rad)))  # tangent
    return np.array(pts, np.float32).reshape(-1, 1, 2), np.array(dirs, np.float32)


_TEMPLATE, _TDIRS = _template_points()
# outer wires count more: a tiny board squeezed onto the dense centre must not look good
_TW = (
    np.hypot(_TEMPLATE[:, 0, 0] - board.CENTER[0], _TEMPLATE[:, 0, 1] - board.CENTER[1])
    / board.DOUBLE_OUTER_PX
).astype(np.float32)
_TW /= _TW.sum()
N_BINS = 8


# ---------- helpers ----------
def _work_image(frame: np.ndarray) -> tuple[np.ndarray, float]:
    f = min(1.0, WORK_WIDTH / frame.shape[1])
    img = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1.0 else frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.GaussianBlur(gray, (5, 5), 0), f


def bull_candidates(gray: np.ndarray, top: int = 3) -> list[tuple[float, float]]:
    """Points where many straight edge segments meet (the spokes meet at the bull)."""
    med = float(np.median(gray))
    edges = cv2.Canny(gray, max(10, 0.5 * med), max(30, 1.3 * med))
    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 25, minLineLength=max(25, gray.shape[1] // 18), maxLineGap=6
    )
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float64)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-length)[:300]
    segs, length = segs[order], length[order]
    p, d = segs[:, :2], segs[:, 2:] - segs[:, :2]
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    n = len(segs)
    i, j = np.triu_indices(n, 1)
    cross = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
    ok = np.abs(cross) > math.sin(math.radians(10))
    i, j, cross = i[ok], j[ok], cross[ok]
    if len(i) == 0:
        return []
    t = ((p[j, 0] - p[i, 0]) * d[j, 1] - (p[j, 1] - p[i, 1]) * d[j, 0]) / cross
    pt = p[i] + d[i] * t[:, None]
    h, w = gray.shape
    inb = (pt[:, 0] >= 0) & (pt[:, 0] < w) & (pt[:, 1] >= 0) & (pt[:, 1] < h)
    pt, wgt = pt[inb], np.minimum(length[i], length[j])[inb]
    cell = 6
    hist = np.zeros((h // cell + 2, w // cell + 2), np.float32)
    np.add.at(hist, ((pt[:, 1] // cell).astype(int), (pt[:, 0] // cell).astype(int)), wgt)
    hist = cv2.GaussianBlur(hist, (0, 0), 1.2)
    out = []
    for _ in range(top):
        y, x = np.unravel_index(int(hist.argmax()), hist.shape)
        if hist[y, x] <= 0:
            break
        out.append(((x + 0.5) * cell, (y + 0.5) * cell))
        hist[max(0, y - 6) : y + 7, max(0, x - 6) : x + 7] = 0
    return out


def _init_H(center, semi: float, ratio: float, ang: float, phi_deg: float) -> np.ndarray:
    """Canvas -> image affine map: double-wire circle -> ellipse (semi-major ``semi``)."""
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    cp, sp = math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))
    rot_e = np.array([[ca, -sa], [sa, ca]])
    rot_p = np.array([[cp, -sp], [sp, cp]])
    m = rot_e @ np.diag([semi, semi * ratio]) @ rot_p / board.DOUBLE_OUTER_PX
    H = np.eye(3)
    H[:2, :2] = m
    H[:2, 2] = np.array(center) - m @ np.array(board.CENTER)
    return H


def _up_deviation_deg(H_ci: np.ndarray) -> float:
    p = cv2.perspectiveTransform(
        np.array([[board.CENTER, (board.CENTER[0], board.CENTER[1] - 100)]], np.float32), H_ci
    )[0]
    v = p[1] - p[0]
    return math.degrees(math.atan2(v[0], -v[1]))  # 0 = straight up in the image


def _wedge_points() -> np.ndarray:
    pts = [
        board.polar_to_canvas(r * board.PX_PER_MM, i * 18.0 + d)
        for i in range(20)
        for r in (122, 134, 148)
        for d in (-3.0, 0.0, 3.0)
    ]
    return np.array(pts, np.float32).reshape(-1, 1, 2)


_WEDGES = _wedge_points()
_ALT = np.array([1.0 if i % 2 else -1.0 for i in range(20)])


def _alternation_score(gray: np.ndarray, H_ci: np.ndarray) -> float:
    """Correlation of the 20 wedge brightnesses with the black/cream alternation."""
    q = cv2.perspectiveTransform(_WEDGES, H_ci).reshape(-1, 2)
    xs = np.clip(q[:, 0].round().astype(int), 0, gray.shape[1] - 1)
    ys = np.clip(q[:, 1].round().astype(int), 0, gray.shape[0] - 1)
    g = gray[ys, xs].astype(np.float64).reshape(20, 9).mean(axis=1)
    return float(((g - g.mean()) * _ALT).sum() / (g.std() * 20 + 1e-6))


class _Chamfer:
    """Oriented chamfer matching: a wire only matches edges running the same way."""

    def __init__(self, gray: np.ndarray) -> None:
        med = float(np.median(gray))
        edges = cv2.Canny(gray, max(10, 0.5 * med), max(30, 1.3 * med))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        theta = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0  # edge direction, 0..180
        b = np.floor(theta / (180.0 / N_BINS)).astype(int) % N_BINS
        on = edges > 0
        self.dt = np.empty((N_BINS,) + gray.shape, np.float32)
        for k in range(N_BINS):
            near = on & ((b == k) | (b == (k + 1) % N_BINS) | (b == (k - 1) % N_BINS))
            self.dt[k] = cv2.distanceTransform(
                np.where(near, 0, 255).astype(np.uint8), cv2.DIST_L2, 3
            )
        self.h, self.w = gray.shape

    def _dists(self, H_ci: np.ndarray, trunc: float) -> np.ndarray:
        q = cv2.perspectiveTransform(_TEMPLATE, H_ci).reshape(-1, 2)
        q2 = cv2.perspectiveTransform(_TEMPLATE + _TDIRS.reshape(-1, 1, 2) * 3.0, H_ci).reshape(
            -1, 2
        )
        ang = np.degrees(np.arctan2(q2[:, 1] - q[:, 1], q2[:, 0] - q[:, 0])) % 180.0
        bins = np.floor(ang / (180.0 / N_BINS)).astype(int) % N_BINS
        x, y = q[:, 0].round().astype(int), q[:, 1].round().astype(int)
        inside = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
        d = np.full(len(q), trunc, np.float32)
        d[inside] = np.minimum(self.dt[bins[inside], y[inside], x[inside]], trunc)
        return d

    def cost(self, H_ci: np.ndarray, trunc: float) -> float:
        return float((self._dists(H_ci, trunc) * _TW).sum())

    def inliers(self, H_ci: np.ndarray, tol: float) -> tuple[float, float]:
        d = self._dists(H_ci, 30.0)
        return float(((d <= tol) * _TW).sum()), float((d * _TW).sum())


def _anchors_to_H(img4: np.ndarray) -> Optional[np.ndarray]:
    try:
        return cv2.getPerspectiveTransform(_CANVAS_ANCHORS, img4.astype(np.float32))
    except cv2.error:
        return None


def _quad_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


MAX_DRIFT = 0.22  # anchors may move this fraction of their own radius away from the start
MIN_AREA_FRAC = 0.6  # ...and the anchor quad must keep this much of its starting area


def _refine(
    ch: _Chamfer, H_ci: np.ndarray, steps=None, drift: float = MAX_DRIFT
) -> tuple[Optional[np.ndarray], float]:
    """Pattern search over the four outer-wire anchor positions (bounded, so it cannot collapse)."""
    p0 = (
        cv2.perspectiveTransform(_CANVAS_ANCHORS.reshape(-1, 1, 2), H_ci)
        .reshape(4, 2)
        .astype(np.float64)
    )
    area0 = _quad_area(p0)
    radius = math.sqrt(area0 / 2.0)  # a circle's inscribed square has area 2 r^2
    p = p0.copy()

    def eval_(pp, trunc):
        if np.abs(pp - p0).max() > drift * radius or _quad_area(pp) < MIN_AREA_FRAC * area0:
            return 1e9, None
        H = _anchors_to_H(pp)
        if H is None or abs(np.linalg.det(H)) < 1e-12:
            return 1e9, None
        return ch.cost(H, trunc), H

    best_cost, best_H = eval_(p, 14.0)
    for step in steps or SEARCH_STEPS:
        step = step * radius / 100.0 if radius > 0 else step
        trunc = max(3.0, min(14.0, step * 2.0))
        best_cost, best_H = eval_(p, trunc)
        improved, guard = True, 0
        while improved and guard < 40:
            improved = False
            guard += 1
            for i in range(4):
                for j in range(2):
                    for sgn in (-1.0, 1.0):
                        q = p.copy()
                        q[i, j] += sgn * step
                        c, H = eval_(q, trunc)
                        if c < best_cost - 1e-6:
                            p, best_cost, best_H, improved = q, c, H, True
    final_cost, final_H = eval_(p, 3.0)
    return final_H, final_cost


# ---------- public API ----------
SEMI_FRACS = tuple(np.geomspace(0.10, 0.55, 11))  # double-wire semi-major axis / image width
RATIOS = (0.45, 0.6, 0.75, 0.9)  # minor / major axis
ANGLES = tuple(range(0, 180, 20))  # ellipse orientation
N_REFINE = 5
POLISH_STEPS = (1.5, 0.8, 0.4, 0.2)  # percent of the board radius, at full resolution
POLISH_DRIFT = 0.06


def _fit_around(ch: _Chamfer, gray: np.ndarray, center) -> Optional[tuple[float, np.ndarray]]:
    w = gray.shape[1]
    hyps = []
    for ang in ANGLES:
        for ratio in RATIOS:
            phis = [
                float(p)
                for p in range(0, 360, ORIENT_STEP_DEG)
                if abs(_up_deviation_deg(_init_H(center, 100.0, ratio, ang, float(p))))
                <= MAX_UP_DEVIATION_DEG
            ]
            for fr in SEMI_FRACS:
                semi = fr * w
                phi = max(
                    phis,
                    key=lambda p: _alternation_score(gray, _init_H(center, semi, ratio, ang, p)),
                )
                H0 = _init_H(center, semi, ratio, ang, phi)
                hyps.append((ch.cost(H0, 8.0), H0))
    hyps.sort(key=lambda t: t[0])
    best = None
    for _, H0 in hyps[:N_REFINE]:
        H, cost = _refine(ch, H0)
        if H is None:
            continue
        score = ch.inliers(H, 1.6)[0]
        if best is None or score > best[0]:
            best = (score, H)
    return best


def detect_board(
    frame: np.ndarray, bull: Optional[tuple[float, float]] = None
) -> Optional[AutoResult]:
    """Find the board in a BGR frame. ``bull`` (x, y in frame px) skips the bull search.

    Returns None if nothing board-like is found.
    """
    gray, f = _work_image(frame)
    ch = _Chamfer(gray)
    centers = [(bull[0] * f, bull[1] * f)] if bull is not None else bull_candidates(gray)
    best = None
    for c in centers:
        r = _fit_around(ch, gray, c)
        if r is not None and (best is None or r[0] > best[0]):
            best = r
    if best is None:
        return None
    H_ci = best[1]
    conf, mean_d = ch.inliers(H_ci, 1.6)
    S = np.diag([f, f, 1.0])  # original px -> working px
    H_full = np.linalg.inv(S) @ H_ci  # canvas -> original px
    if f < 1.0:  # polish at full resolution for the last pixels
        g_full = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame, (5, 5), 0
        )
        H2, _ = _refine(_Chamfer(g_full), H_full, steps=POLISH_STEPS, drift=POLISH_DRIFT)
        if H2 is not None:
            H_full = H2
    H_cam = np.linalg.inv(H_full)  # original px -> canvas
    return AutoResult(H=H_cam, confidence=conf, mean_edge_px=mean_d / f)
