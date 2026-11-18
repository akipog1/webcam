"""Interactive homography calibration with save/load and reprojection overlay."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import autocal, board

log = logging.getLogger(__name__)

AUTO_MIN_CONFIDENCE = 0.65   # below this an automatic fit is not offered
AUTO_SETTLE_FRAMES = 20      # let the camera's exposure settle before the first automatic attempt


class Calibrator:
    """Maps raw camera pixels to the flat board canvas.

    Click the named targets in order (bull + 4 double wires by default; at
    least 4 required). With >4 points RANSAC rejects a badly clicked point.
    """

    def __init__(self, path: str = "calibration.json") -> None:
        self.path = Path(path)
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.targets = board.calibration_targets()
        self.frame_size: Optional[tuple[int, int]] = None  # (width, height) at calibration time

    # ---------- math ----------
    def fit(self, src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> float:
        """Fit the homography; returns mean reprojection error in canvas px."""
        if len(src) < 4:
            raise ValueError("need at least 4 points")
        s = np.array(src, np.float32)
        d = np.array(dst, np.float32)
        method = cv2.RANSAC if len(src) > 4 else 0
        H, _ = cv2.findHomography(s, d, method, 8.0)
        if H is None:
            raise ValueError("homography failed - points collinear?")
        self.set_matrix(H)
        proj = cv2.perspectiveTransform(s.reshape(-1, 1, 2), H).reshape(-1, 2)
        return float(np.linalg.norm(proj - d, axis=1).mean())

    def set_matrix(self, H: np.ndarray) -> None:
        self.H = H
        self.H_inv = np.linalg.inv(H)

    def to_board(self, pt: tuple[float, float]) -> tuple[float, float]:
        p = cv2.perspectiveTransform(np.array([[pt]], np.float32), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), self.H_inv).reshape(-1, 2)

    # ---------- persistence ----------
    def save(self) -> None:
        data = {"H": self.H.tolist(), "canvas": board.CANVAS_SIZE}
        if self.frame_size is not None:
            data["frame"] = list(self.frame_size)
        self.path.write_text(json.dumps(data, indent=2))
        log.info("calibration saved to %s", self.path)

    def load(self, frame_size: Optional[tuple[int, int]] = None) -> bool:
        """Load a saved calibration. If ``frame_size`` is given and the file
        records a different one, reject it: the homography is in pixel units."""
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text())
            if data.get("canvas") != board.CANVAS_SIZE:
                log.warning("calibration canvas size mismatch; recalibrate")
                return False
            saved = data.get("frame")
            if saved is None:
                log.warning("calibration has no frame size recorded; assuming it matches")
            elif frame_size is not None and tuple(saved) != tuple(frame_size):
                log.warning("calibration was made at %s but camera is %s; recalibrate",
                            tuple(saved), tuple(frame_size))
                return False
            self.set_matrix(np.array(data["H"], np.float64))
            self.frame_size = tuple(saved) if saved is not None else frame_size
            log.info("calibration loaded from %s", self.path)
            return True
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as e:
            log.warning("bad calibration file: %s", e)
            return False

    # ---------- drawing ----------
    def draw_overlay(self, frame: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> None:
        """Reproject the ideal board onto the live frame for visual checking."""
        if self.H_inv is None:
            return
        for r in board.RING_RADII_PX:
            ring = np.array([board.polar_to_canvas(r, a) for a in range(0, 360, 4)], np.float32)
            cv2.polylines(frame, [self.to_camera(ring).astype(np.int32)], True, color, 1, cv2.LINE_AA)
        for i in range(20):
            a = board.sector_wire_angle(i)
            line = np.array([board.polar_to_canvas(board.OUTER_BULL_PX, a),
                             board.polar_to_canvas(board.DOUBLE_OUTER_PX, a)], np.float32)
            p = self.to_camera(line).astype(np.int32)
            cv2.line(frame, tuple(p[0]), tuple(p[1]), color, 1, cv2.LINE_AA)

    # ---------- automatic ----------
    def auto_fit(self, frame: np.ndarray, bull: Optional[tuple[float, float]] = None,
                 min_confidence: float = AUTO_MIN_CONFIDENCE) -> Optional[float]:
        """Detect the board in ``frame`` and set the homography. Returns the detection
        confidence (0..1) if it reached ``min_confidence``, otherwise None (state unchanged)."""
        res = autocal.detect_board(frame, bull=bull)
        if res is None or res.confidence < min_confidence:
            log.info("auto-calibration found no confident fit (%s)",
                     "nothing board-like" if res is None else f"confidence {res.confidence:.2f}")
            return None
        self.set_matrix(res.H)
        self.frame_size = (frame.shape[1], frame.shape[0])
        log.info("auto-calibration: confidence %.2f, mean edge distance %.1f px", res.confidence, res.mean_edge_px)
        return res.confidence

    def target_pixels(self) -> list[tuple[float, float]]:
        """Camera positions of the manual-calibration targets under the current homography."""
        pts = np.array([self.targets[n] for n in self.targets], np.float32)
        return [(float(x), float(y)) for x, y in self.to_camera(pts)]

    # ---------- interactive ----------
    def run_interactive(self, read_frame, auto: bool = True, window_rect=None) -> bool:
        """Collect clicks on live video. Keys: Enter=accept, r=reset, u=undo, a=auto-detect, q=quit.

        With ``auto`` the board is detected automatically after a moment and the 5 points are
        pre-filled; check the overlay and press Enter. If auto fails, click the bull and press
        ``a`` to let the detector do the rest, or click all points by hand.
        """
        names = list(self.targets)
        clicks: list[tuple[float, float]] = []
        win = "Calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if window_rect is not None:
            try:
                x, y, w, h = window_rect
                cv2.resizeWindow(win, int(w), int(h))
                cv2.moveWindow(win, int(x), int(y))
            except cv2.error:
                pass
        cv2.setMouseCallback(win, lambda e, x, y, *_: clicks.append((x, y))
                             if e == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(names) else None)
        err: Optional[float] = None
        note = ""
        misses = frames = 0
        auto_pending = auto
        want_auto = False
        while True:
            frame = read_frame()
            if frame is None:
                misses += 1
                if misses > 100:
                    log.error("no frames available; calibration aborted")
                    cv2.destroyWindow(win)
                    return False
                cv2.waitKey(10)
                continue
            misses = 0
            frames += 1
            self.frame_size = (frame.shape[1], frame.shape[0])
            if (auto_pending and frames >= AUTO_SETTLE_FRAMES) or want_auto:
                hint = "Detecting the board..."
                wait = frame.copy()
                cv2.putText(wait, hint, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(win, wait)
                cv2.waitKey(1)
                bull = clicks[0] if (want_auto and clicks) else None
                conf = self.auto_fit(frame, bull=bull, min_confidence=0.5 if bull else AUTO_MIN_CONFIDENCE)
                if conf is not None:
                    clicks[:] = self.target_pixels()
                    note = f"Auto-detected (confidence {conf:.2f}) - check the rings, Enter to accept"
                else:
                    note = ("Auto failed - click the bull then press a, or click the 5 points"
                            if not bull else "Still no fit - click all 5 points by hand")
                auto_pending = want_auto = False
            disp = frame.copy()
            for i, p in enumerate(clicks):
                q = (int(p[0]), int(p[1]))
                cv2.circle(disp, q, 5, (0, 0, 255), -1)
                cv2.putText(disp, str(i + 1), (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if len(clicks) >= 4:
                err = self.fit(clicks, [self.targets[n] for n in names[:len(clicks)]])
                self.draw_overlay(disp)
            msg = (f"Click: {names[len(clicks)]}" if len(clicks) < len(names) else "All points set")
            cv2.putText(disp, note or msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            footer = "Enter=accept (>=4 pts)  a=auto  u=undo  r=reset  q=quit"
            if err is not None:
                footer = f"err {err:.1f}px  " + footer
            cv2.putText(disp, footer, (20, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (13, 10) and len(clicks) >= 4:
                cv2.destroyWindow(win)
                self.save()
                return True
            if k == ord("r"):
                clicks.clear(); err = None; note = ""
            elif k == ord("u") and clicks:
                clicks.pop(); note = ""
            elif k == ord("a"):
                want_auto = True
            elif k == ord("q"):
                cv2.destroyWindow(win)
                return False
