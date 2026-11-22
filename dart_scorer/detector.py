"""Dart detection with a per-dart state machine.

States::

    WAIT_BASELINE -> WAIT_THROW -> SETTLING -> SCORED -> WAIT_THROW (next dart)
                                                    \\-> WAIT_REMOVAL (after 3 darts)
    WAIT_REMOVAL -> WAIT_THROW once the board matches the turn-start baseline again.

After each scored dart the *reference* frame is replaced by the current frame,
so the next diff only contains the newest dart. Large-area changes (a hand or
body) are ignored and restart settling.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import DetectorConfig

log = logging.getLogger(__name__)


class State(enum.Enum):
    WAIT_BASELINE = "WAIT_BASELINE"
    WAIT_THROW = "WAIT_THROW"
    SETTLING = "SETTLING"
    SCORED = "SCORED"
    WAIT_REMOVAL = "WAIT_REMOVAL"


@dataclass
class Detection:
    """A settled dart, in raw camera pixels."""

    tip: tuple[float, float]
    centroid: tuple[float, float]
    contour: np.ndarray
    area: float


class DartDetector:
    """Feed frames via :meth:`update`; returns a Detection exactly once per dart."""

    def __init__(self, cfg: DetectorConfig, darts_per_turn: int = 3) -> None:
        self.cfg = cfg
        self.darts_per_turn = darts_per_turn
        self.state = State.WAIT_BASELINE
        self.reference: Optional[np.ndarray] = None  # float32, last-dart reference
        self.turn_baseline: Optional[np.ndarray] = None  # empty-board reference
        self.reference_frame: Optional[np.ndarray] = None
        self.turn_baseline_frame: Optional[np.ndarray] = None
        self.darts_this_turn = 0
        self._settle_count = 0
        self._last_mask: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_detection: Optional[Detection] = None
        kernel_size = self.cfg.morph_kernel
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        self._idle_mask: Optional[np.ndarray] = None
        self._idle_mask_shape: Optional[tuple[int, int]] = None

    # ---------- preprocessing ----------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR (or grey) frame -> blurred single channel float32."""
        if frame.ndim == 3:
            if self.cfg.shadow_v_only:
                ch = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
            else:
                ch = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            ch = frame
        k = self.cfg.blur_kernel | 1
        return cv2.GaussianBlur(ch, (k, k), 0).astype(np.float32)

    def diff_mask(self, img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(img, ref).astype(np.uint8)
        _, m = cv2.threshold(diff, self.cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        changed = cv2.countNonZero(m)
        if changed < self.cfg.min_area or changed > self.cfg.hand_motion_area:
            return m
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._morph_kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)
        return m

    # ---------- control ----------
    def capture_baseline(self, frame: np.ndarray) -> None:
        img = self.preprocess(frame)
        self.reference = img.copy()
        self.turn_baseline = img.copy()
        self.reference_frame = frame.copy()
        self.turn_baseline_frame = frame.copy()
        self.darts_this_turn = 0
        self._settle_count = 0
        self.state = State.WAIT_THROW
        log.info("baseline captured")

    def next_turn(self, frame: np.ndarray) -> None:
        """Manual turn reset (key ``n``): treat the current board as empty."""
        self.capture_baseline(frame)

    def undo_dart(self, frame: np.ndarray) -> None:
        """Roll back one dart after a manual undo.

        Leaves WAIT_REMOVAL/SCORED/SETTLING so the next throw is detected, and
        re-bases the reference on the current frame (the dart may still be in
        the board). The turn-start baseline is kept for removal detection.
        """
        self.darts_this_turn = max(0, self.darts_this_turn - 1)
        self._settle_count = 0
        self._last_mask = None
        if self.state != State.WAIT_BASELINE:
            self.reference = self.preprocess(frame)
            self.reference_frame = frame.copy()
            self.state = State.WAIT_THROW

    # ---------- main ----------
    def update(self, frame: np.ndarray) -> Optional[Detection]:
        if self.state == State.WAIT_BASELINE or self.reference is None:
            return None
        baseline_frame = (
            self.turn_baseline_frame
            if self.state == State.WAIT_REMOVAL
            else self.reference_frame
        )
        if baseline_frame is not None and np.array_equal(frame, baseline_frame):
            shape = (frame.shape[0], frame.shape[1])
            if self._idle_mask_shape != shape:
                self._idle_mask = np.zeros(shape, np.uint8)
                self._idle_mask_shape = shape
            self.last_mask = self._idle_mask
            if self.state == State.WAIT_REMOVAL:
                self._settle_count += 1
                if self._settle_count >= self.cfg.settle_frames:
                    log.info("darts removed; ready for next turn")
                    self.capture_baseline(frame)
            else:
                self._settle_count = 0
                self._last_mask = None
                self.state = State.WAIT_THROW
            return None
        img = self.preprocess(frame)

        if self.state == State.WAIT_REMOVAL:
            m = self.diff_mask(img, self.turn_baseline)
            self.last_mask = m
            if cv2.countNonZero(m) < self.cfg.min_area * 2:
                self._settle_count += 1
                if self._settle_count >= self.cfg.settle_frames:
                    log.info("darts removed; ready for next turn")
                    self.capture_baseline(frame)
            else:
                self._settle_count = 0
            return None

        mask = self.diff_mask(img, self.reference)
        self.last_mask = mask
        changed = cv2.countNonZero(mask)

        if changed > self.cfg.hand_motion_area:
            self._settle_count = 0
            self._last_mask = None
            self.state = State.SETTLING
            return None

        if changed < self.cfg.min_area:
            # idle: absorb slow lighting drift
            cv2.accumulateWeighted(img, self.reference, self.cfg.background_alpha)
            self.state = State.WAIT_THROW
            self._settle_count = 0
            self._last_mask = None
            return None

        # something changed and is small enough: settle
        if self._last_mask is not None:
            delta = cv2.countNonZero(cv2.bitwise_xor(mask, self._last_mask))
            stable = delta <= max(self.cfg.min_area, self.cfg.stable_pixel_tolerance * changed)
            self._settle_count = self._settle_count + 1 if stable else 0
        self._last_mask = mask
        self.state = State.SETTLING
        if self._settle_count < self.cfg.settle_frames:
            return None

        det = self._find_dart(mask)
        self._settle_count = 0
        self._last_mask = None
        if det is None:
            # a stable change that isn't a dart (e.g. lighting); absorb it
            self.reference = img.copy()
            self.state = State.WAIT_THROW
            return None
        self.reference = img.copy()  # only the next dart will differ
        self.reference_frame = frame.copy()
        self.darts_this_turn += 1
        self.last_detection = det
        self.state = (
            State.WAIT_REMOVAL if self.darts_this_turn >= self.darts_per_turn else State.SCORED
        )
        if self.state == State.WAIT_REMOVAL:
            self._settle_count = 0
        return det

    def after_scored(self) -> None:
        """Called by App once the SCORED state has been displayed."""
        if self.state == State.SCORED:
            self.state = State.WAIT_THROW

    # ---------- contour analysis ----------
    def _find_dart(self, mask: np.ndarray) -> Optional[Detection]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[tuple[float, np.ndarray]] = None
        for c in contours:
            area = cv2.contourArea(c)
            if not self.cfg.min_area <= area <= self.cfg.max_area:
                continue
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            if min(w, h) < 1:
                continue
            aspect = max(w, h) / min(w, h)
            hull = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull if hull else 0
            if aspect < self.cfg.min_aspect_ratio or solidity < self.cfg.min_solidity:
                continue
            if best is None or area > best[0]:
                best = (area, c)
        if best is None:
            return None
        area, c = best
        mom = cv2.moments(c)
        cen = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]) if mom["m00"] else tuple(c[0][0])
        return Detection(estimate_tip(c, self.cfg.tip_method, self.cfg.camera_side), cen, c, area)


def estimate_tip(
    contour: np.ndarray, method: str = "fitline", camera_side: str = "bottom"
) -> tuple[float, float]:
    """Estimate the dart tip in raw image pixels.

    Assumptions:
      * ``centroid``: board-normal camera; dart appears as a dot. Least accurate.
      * ``lowest``: camera mounted below/in front looking up at the board; the tip
        is the point of the contour closest to ``camera_side`` of the image.
      * ``fitline`` (default): fit a line along the dart barrel/flight and take the
        extreme contour point along that line which lies toward ``camera_side``.
        Darts lean away from the camera, so the tip (stuck in the board) is the
        end nearest that edge. Works for side-mounted cameras with ``left``/``right``.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    side_vec = {"bottom": (0, 1), "top": (0, -1), "left": (-1, 0), "right": (1, 0)}[camera_side]
    if method == "centroid":
        m = pts.mean(axis=0)
        return float(m[0]), float(m[1])
    if method == "lowest":
        i = int(np.argmax(pts @ np.array(side_vec, np.float32)))
        return float(pts[i, 0]), float(pts[i, 1])
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_WELSCH, 0, 0.01, 0.01).flatten()
    d = np.array([vx, vy], np.float32)
    if d @ np.array(side_vec, np.float32) < 0:
        d = -d
    proj = (pts - np.array([x0, y0], np.float32)) @ d
    # Avoid letting a few isolated mask pixels define the dart's endpoint.
    p = np.array([x0, y0], np.float32) + d * np.percentile(proj, 98.0)
    return float(p[0]), float(p[1])
