"""Live diagnostic dashboard for single- and dual-camera dart scoring.

The displayed shape confidence is a heuristic quality indicator, not a
statistically calibrated probability. Camera skew is measured from sequential
frame-read completion times; it is not hardware-trigger synchronisation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import board
from .detector import DartDetector, Detection
from .scorer import score_point


@dataclass
class CameraStats:
    fps: float = 0.0
    processing_ms: float = 0.0
    frames: int = 0
    detections: int = 0
    last_detection: Optional[Detection] = None
    confidence: Optional[float] = None
    last_update: Optional[float] = None
    last_update: Optional[float] = None


def _text(img, value: str, xy, scale=0.52, color=(245, 245, 245), thickness=1):
    cv2.putText(img, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fit(frame: Optional[np.ndarray], size=(480, 270)) -> np.ndarray:
    w, h = size
    out = np.zeros((h, w, 3), np.uint8)
    if frame is None:
        _text(out, "No camera frame", (15, 35))
        return out
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh))
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y+nh, x:x+nw] = resized
    return out


def _quality(det: Optional[Detection], detector: DartDetector) -> Optional[float]:
    if det is None:
        return None
    rect = cv2.minAreaRect(det.contour)
    a, b = rect[1]
    aspect = max(a, b) / max(1e-6, min(a, b))
    hull = cv2.contourArea(cv2.convexHull(det.contour))
    solidity = det.area / hull if hull else 0.0
    cfg = detector.cfg
    aspect_q = min(1.0, aspect / max(cfg.min_aspect_ratio * 2, 1.0))
    solidity_q = min(1.0, solidity / max(cfg.min_solidity, 1e-6))
    area_mid = (cfg.min_area + cfg.max_area) / 2
    area_q = max(0.0, 1.0 - abs(det.area - area_mid) / max(area_mid, 1.0))
    return round(100 * (0.5 * aspect_q + 0.3 * solidity_q + 0.2 * area_q), 1)


class LiveDashboard:
    """Renders camera feeds, board projection and runtime diagnostic metrics."""
    WINDOW = "Dart Scorer - Live Diagnostics"

    def __init__(self, dual: bool = False) -> None:
        self.dual = dual
        self.stats = [CameraStats(), CameraStats()]
        self.false_positives = 0
        self.missed_darts = 0
        self._last_score = "Waiting for dart"
        self._last_method = "-"
        self._last_board = None
        self._read_times = [None, None]
        self._latency_ms = 0.0

    def record_frame(self, index: int, read_time: float, processing_ms: float,
                     detector: DartDetector, detection: Optional[Detection]) -> None:
        s = self.stats[index]
        now = time.perf_counter()
        dt = now - s.last_update if s.last_update is not None else 0.0
        if dt > 0:
            instantaneous = 1.0 / dt
            s.fps = instantaneous if s.fps == 0 else 0.85 * s.fps + 0.15 * instantaneous
        s.last_update = now
        s.frames += 1
        s.processing_ms = processing_ms
        self._read_times[index] = read_time
        if detection is not None:
            s.detections += 1
            s.last_detection = detection
            s.confidence = _quality(detection, detector)
            self._latency_ms = processing_ms

    def record_result(self, flat, method: str = "single-camera") -> None:
        self._last_board = tuple(flat)
        self._last_score = score_point(*flat).label
        self._last_method = method

    def render(self, frame1, frame2, det1: DartDetector, det2: Optional[DartDetector],
               cal1, cal2, game, controller, sync_skew_ms: Optional[float] = None) -> np.ndarray:
        cam1 = _fit(frame1)
        d1 = det1.last_detection
        if d1 is not None:
            scale = min(480 / frame1.shape[1], 270 / frame1.shape[0])
            ox = int((480 - frame1.shape[1] * scale) / 2)
            oy = int((270 - frame1.shape[0] * scale) / 2)
            cv2.circle(cam1, (int(d1.tip[0] * scale + ox), int(d1.tip[1] * scale + oy)), 7, (0, 0, 255), 2)
            _text(cam1, f"TIP {d1.tip[0]:.1f}, {d1.tip[1]:.1f}", (10, 255), .48, (0, 255, 255))
        _text(cam1, f"CAM 1 | {det1.state.value}", (10, 22), .55)
        view2 = frame2.copy() if frame2 is not None else None
        if view2 is not None and cal2 is not None:
            cal2.draw_overlay(view2, (0, 200, 200))
        cam2 = _fit(view2)
        if det2 is not None:
            d2 = det2.last_detection
            if d2 is not None and frame2 is not None:
                scale = min(480 / frame2.shape[1], 270 / frame2.shape[0])
                ox = int((480 - frame2.shape[1] * scale) / 2)
                oy = int((270 - frame2.shape[0] * scale) / 2)
                cv2.circle(cam2, (int(d2.tip[0] * scale + ox), int(d2.tip[1] * scale + oy)), 7, (0, 0, 255), 2)
                _text(cam2, f"TIP {d2.tip[0]:.1f}, {d2.tip[1]:.1f}", (10, 255), .48, (0, 255, 255))
            _text(cam2, f"CAM 2 | {det2.state.value}", (10, 22), .55)
        else:
            _text(cam2, "CAM 2 | not enabled", (10, 22), .55)

        board_img = np.zeros((400, 400, 3), np.uint8)
        cv2.circle(board_img, (200, 200), 185, (130, 130, 130), 1)
        for r in board.RING_RADII_PX:
            cv2.circle(board_img, (200, 200), int(r * 185 / board.DOUBLE_OUTER_PX), (75, 75, 75), 1)
        for i in range(20):
            angle = np.deg2rad(i * 18 - 9)
            end = (int(200 + 185 * np.sin(angle)), int(200 - 185 * np.cos(angle)))
            cv2.line(board_img, (200, 200), end, (55, 55, 55), 1)
        if self._last_board is not None:
            bx, by = self._last_board
            px = int(200 + (bx - board.CENTER[0]) * 185 / board.DOUBLE_OUTER_PX)
            py = int(200 + (by - board.CENTER[1]) * 185 / board.DOUBLE_OUTER_PX)
            cv2.circle(board_img, (px, py), 6, (0, 0, 255), -1)
            cv2.circle(board_img, (px, py), 11, (0, 255, 255), 1)
        _text(board_img, "BOARD POSITION", (10, 20), .55)

        panel = np.zeros((400, 560, 3), np.uint8)
        y = 25
        _text(panel, "LIVE PERFORMANCE", (12, y), .65, (0, 255, 255)); y += 32
        for i, det in enumerate((det1, det2)):
            if det is None:
                continue
            st = self.stats[i]
            _text(panel, f"Camera {i+1}: {st.fps:.1f} FPS | {st.processing_ms:.1f} ms/frame", (12, y)); y += 24
            _text(panel, f"  State: {det.state.value} | detections: {st.detections}", (12, y)); y += 24
            _text(panel, f"  Tip quality heuristic: {st.confidence if st.confidence is not None else '—'} / 100", (12, y)); y += 30
        _text(panel, f"Detector processing (last): {self._latency_ms:.1f} ms", (12, y)); y += 25
        _text(panel, f"Camera read skew: {sync_skew_ms:.1f} ms" if sync_skew_ms is not None else "Camera read skew: n/a", (12, y)); y += 30
        if det2 is not None and det1.last_detection is not None and det2.last_detection is not None:
            p1 = cal1.to_board(det1.last_detection.tip)
            p2 = cal2.to_board(det2.last_detection.tip)
            agreement = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            _text(panel, f"Camera projected-tip separation: {agreement:.1f} board px", (12, y)); y += 25
        else:
            _text(panel, "Camera agreement: awaiting both detections", (12, y)); y += 25
        _text(panel, f"Last result: {self._last_score} via {self._last_method}", (12, y), .55, (0, 255, 0)); y += 25
        if self._last_board is not None:
            _text(panel, f"Board XY: {self._last_board[0]:.1f}, {self._last_board[1]:.1f}", (12, y)); y += 24
        _text(panel, f"Game: {game.player.name} | remaining {game.player.remaining}", (12, y)); y += 25
        _text(panel, f"False positives (manual): {self.false_positives}   Missed darts (manual): {self.missed_darts}", (12, y), .43); y += 25
        _text(panel, "Keys: X = false positive   M = missed dart   D = toggle dashboard", (12, y), .42)
        _text(panel, "Tip quality is heuristic, NOT a calibrated probability.", (12, 375), .40, (0, 190, 255))

        top = np.hstack((cam1, cam2))
        bottom = np.hstack((cv2.resize(board_img, (400, 400)), panel))
        return np.vstack((top, bottom))
