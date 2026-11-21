"""Main application: camera/video loop, HUD and windows. Logic lives in controller.py."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import board
from .calibration import Calibrator
from .config import AppConfig, parse_args
from .controller import Action, Controller, ThrowLogger  # ThrowLogger re-exported
from .detector import DartDetector
from .fusion import DetectionPairer, fuse
from .game import GameState
from .layout import apply_layout, compute_layout, screen_size
from .sources import Camera, Recorder, VideoFile, open_source, open_source2  # Camera re-exported
from .live_dashboard import LiveDashboard
import time

log = logging.getLogger("dart_scorer")

BOARD_VIEW_PX = 600


class App:
    """Ties source, calibration, detector, game and HUD together.

    Keys: b=baseline  n=next turn  u=undo  c=recalibrate  e=type override
    (0-9/t/d/s/b + Enter); q=quit. Click in the Board window to re-score the
    last dart at that position.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.cam = open_source(cfg)
        self.cal = Calibrator(cfg.calibration_file)
        self.det = DartDetector(cfg.detector)
        self.game = GameState.new(cfg.players, cfg.game, cfg.double_out)
        self.ctl = Controller(self.game, self.det, self.cal, ThrowLogger(cfg.log_file))
        self.recorder: Optional[Recorder] = None
        # optional second camera: independent detector, fused on each detection
        self.cam2 = self.cal2 = self.det2 = self.pairer = None
        self.recorder2: Optional[Recorder] = None
        self.last_fused = None
        self.dashboard = LiveDashboard(cfg.dual)
        self.dashboard_enabled = True
        self._read_time1 = self._read_time2 = None
        self._dashboard_frame = None
        if cfg.dual:
            self.cam2 = open_source2(cfg)
            self.cal2 = Calibrator(cfg.calibration_file2)
            self.det2 = DartDetector(
                dataclasses.replace(cfg.detector, camera_side=cfg.camera_side2)
            )
            self.pairer = DetectionPairer(cfg.fuse_window)
            self.ctl.det2 = self.det2

    # ---------- calibration ----------
    @staticmethod
    def _calibration_reader(cam):
        """Frames for the calibration window; a clip is looped so you can take your time."""

        def read():
            f = cam.read()
            if f is None and not cam.is_live:
                cam.rewind()
                f = cam.read()
            return f

        return read

    def _calibrate(self, second: bool = False) -> bool:
        cam, cal = (self.cam2, self.cal2) if second else (self.cam, self.cal)
        ok = cal.run_interactive(
            self._calibration_reader(cam),
            auto=self.cfg.auto_calibrate,
            window_rect=self._calibration_rect(),
        )
        if not cam.is_live:
            cam.rewind()
        return ok

    def _layout(self) -> dict:
        return compute_layout(
            screen_size(self.cfg.screen),
            self.cam2 is not None,
            self.cam.size,
            self.cam2.size if self.cam2 is not None else None,
            self.cfg.debug,
        )

    def _calibration_rect(self):
        """Calibration window: the camera-1 tile, or None to leave it where the OS puts it."""
        return self._layout().get("Dart Scorer") if self.cfg.layout else None

    def _record2_path(self) -> str:
        if self.cfg.record2:
            return self.cfg.record2
        p = Path(self.cfg.record)
        return str(p.with_name(p.stem + "_cam2" + p.suffix))

    # ---------- main loop ----------
    def run(self) -> None:
        if not self.cam.is_open():
            log.error("could not open %s", self.cfg.video or f"camera {self.cfg.camera}")
            return
        if self.cam2 is not None and not self.cam2.is_open():
            log.error("could not open %s", self.cfg.video2 or f"camera {self.cfg.camera2}")
            return
        if self.cfg.recalibrate or not self.cal.load(self.cam.size):
            if not self._calibrate():
                log.info("calibration cancelled")
                return
        if self.cam2 is not None and (self.cfg.recalibrate or not self.cal2.load(self.cam2.size)):
            if not self._calibrate(second=True):
                log.info("calibration of camera 2 cancelled")
                return
        if self.cfg.record:
            self.recorder = Recorder(self.cfg.record, self.cam.fps, self.cam.size)
            if self.cam2 is not None:
                self.recorder2 = Recorder(self._record2_path(), self.cam2.fps, self.cam2.size)
        if self.cfg.layout:
            apply_layout(self._layout())
        cv2.namedWindow("Board")
        cv2.setMouseCallback("Board", self._on_board_click)
        delay = 1 if self.cam.is_live else max(1, int(1000 / self.cam.fps))
        try:
            while True:
                frame = self.cam.read()
                self._read_time1 = time.perf_counter()
                if frame is None:
                    log.error("camera lost" if self.cam.is_live else "end of video")
                    break
                if self.recorder:
                    self.recorder.write(frame)
                frame2 = None
                if self.cam2 is not None:
                    frame2 = self.cam2.read()
                    self._read_time2 = time.perf_counter()
                    if frame2 is None:
                        log.error("camera 2 lost" if self.cam2.is_live else "end of video 2")
                        break
                    if self.recorder2:
                        self.recorder2.write(frame2)
                t0 = time.perf_counter()
                det = self.det.update(frame)
                ms1 = (time.perf_counter() - t0) * 1000.0
                self.dashboard.record_frame(0, self._read_time1, ms1, self.det, det)
                det2 = None
                if self.cam2 is None:
                    if det is not None:
                        scored = self.ctl.on_detection(det)
                        if scored is not None:
                            self.dashboard.record_result(self.cal.to_board(det.tip), "cam1")
                else:
                    t0 = time.perf_counter()
                    det2 = self.det2.update(frame2)
                    ms2 = (time.perf_counter() - t0) * 1000.0
                    self.dashboard.record_frame(1, self._read_time2, ms2, self.det2, det2)
                    self._update_dual(det, det2)
                self._draw(frame)
                if frame2 is not None:
                    self._draw2(frame2)
                if self.dashboard_enabled:
                    skew = (
                        abs(self._read_time2 - self._read_time1) * 1000.0
                        if self._read_time1 is not None and self._read_time2 is not None
                        else None
                    )
                    if self.dashboard.should_render():
                        self._dashboard_frame = self.dashboard.render(
                            frame,
                            frame2,
                            self.det,
                            self.det2,
                            self.cal,
                            self.cal2,
                            self.game,
                            self.ctl,
                            skew,
                        )
                    if self._dashboard_frame is not None:
                        cv2.imshow(LiveDashboard.WINDOW, self._dashboard_frame)
                key = cv2.waitKey(delay) & 0xFF
                if self.ctl.edit_buffer is None and key in (ord("d"), ord("D")):
                    self.dashboard_enabled = not self.dashboard_enabled
                    if not self.dashboard_enabled:
                        cv2.destroyWindow(LiveDashboard.WINDOW)
                    continue
                if self.ctl.edit_buffer is None and key in (ord("x"), ord("X")):
                    self.dashboard.false_positives += 1
                    continue
                if self.ctl.edit_buffer is None and key in (ord("m"), ord("M")):
                    self.dashboard.missed_darts += 1
                    continue
                action = self.ctl.handle_key(key, frame, frame2)
                if action is Action.QUIT:
                    break
                if action is Action.RECALIBRATE:
                    ok = self._calibrate()
                    if ok and self.cam2 is not None:
                        ok = self._calibrate(second=True)
                    if ok:
                        self.ctl.after_recalibrate()
                        if self.pairer:
                            self.pairer.clear()
                self.det.after_scored()
                if self.det2 is not None:
                    self.det2.after_scored()
        finally:
            self.cam.release()
            if self.cam2 is not None:
                self.cam2.release()
            if self.recorder:
                self.recorder.release()
            if self.recorder2:
                self.recorder2.release()
            cv2.destroyAllWindows()

    def _update_dual(self, d1, d2) -> None:
        """Pair detections from the two cameras and score the fused result."""
        ready = []
        if d1 is not None:
            ready += self.pairer.push(1, d1)
        if d2 is not None:
            ready += self.pairer.push(2, d2)
        ready += self.pairer.tick()
        for a, b in ready:
            tip = fuse(a, self.cal, b, self.cal2, self.cfg.min_cross_angle)
            if tip is not None:
                self.last_fused = tip
                scored = self.ctl.on_fused(tip)
                if scored is not None:
                    self.dashboard.record_result(tip.flat, tip.method)
                log.info(
                    "fused (%s) -> %s",
                    tip.method,
                    self.game.turn_darts[-1].label if self.game.turn_darts else "-",
                )

    def _draw2(self, frame) -> None:
        disp = frame.copy()
        self.cal2.draw_overlay(disp, (0, 200, 200))
        d = self.det2.last_detection
        if d is not None:
            cv2.circle(disp, (int(d.tip[0]), int(d.tip[1])), 5, (0, 0, 255), -1)
        t = f"cam 2 [{self.det2.state.value}]" + (
            f"  last: {self.last_fused.method}" if self.last_fused else ""
        )
        cv2.putText(disp, t, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(disp, t, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Dart Scorer 2", disp)

    def _board_view_size(self) -> int:
        return board.CANVAS_SIZE if self.cfg.debug else BOARD_VIEW_PX

    def _on_board_click(self, event, x, y, *_) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            k = board.CANVAS_SIZE / self._board_view_size()  # window px -> canvas px
            self.ctl.board_click(x * k, y * k)

    # ---------- HUD ----------
    def _draw(self, frame) -> None:
        disp = frame.copy()
        self.cal.draw_overlay(disp, (0, 200, 200))
        d = self.det.last_detection
        if d is not None and self.cfg.debug:
            cv2.drawContours(disp, [d.contour], -1, (0, 255, 0), 1)
        if d is not None:
            cv2.circle(disp, (int(d.tip[0]), int(d.tip[1])), 5, (0, 0, 255), -1)
        g, p, c = self.game, self.game.player, self.ctl
        lines = [
            f"{p.name}: {p.remaining}   avg {p.three_dart_average:.1f}",
            f"Darts: {' '.join(s.label for s in g.turn_darts) or '-'}   turn {g.turn_total}",
            f"[{self.det.state.value}{'/' + self.det2.state.value if self.det2 else ''}] {c.message}",
        ]
        if c.edit_buffer is not None:
            lines.append(f"Override: {c.edit_buffer}_")
        if self.recorder:
            lines.append("REC")
        for i, t in enumerate(lines):
            cv2.putText(disp, t, (20, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(
                disp, t, (20, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
        others = "  ".join(f"{q.name}:{q.remaining}" for q in g.players)
        cv2.putText(
            disp, others, (20, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.imshow("Dart Scorer", disp)

        warped = cv2.warpPerspective(frame, self.cal.H, (board.CANVAS_SIZE, board.CANVAS_SIZE))
        draw_board_lines(warped)
        for pos, s in c.marks:
            if pos is None:
                continue
            x, y = pos
            cv2.circle(warped, (int(x), int(y)), 7, (0, 0, 255), -1)
            cv2.putText(
                warped,
                s.label,
                (int(x) + 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        cv2.imshow(
            "Board", cv2.resize(warped, (BOARD_VIEW_PX,) * 2) if not self.cfg.debug else warped
        )
        if self.cfg.debug and self.det.last_mask is not None:
            cv2.imshow("Mask", self.det.last_mask)


def draw_board_lines(img: np.ndarray, color=(0, 255, 255)) -> None:
    c = (int(board.CENTER[0]), int(board.CENTER[1]))
    for r in board.RING_RADII_PX:
        cv2.circle(img, c, int(round(r)), color, 1, cv2.LINE_AA)
    for i in range(20):
        a = board.sector_wire_angle(i)
        p1 = tuple(int(v) for v in board.polar_to_canvas(board.OUTER_BULL_PX, a))
        p2 = tuple(int(v) for v in board.polar_to_canvas(board.DOUBLE_OUTER_PX, a))
        cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
        lx, ly = board.polar_to_canvas(board.DOUBLE_OUTER_PX + 30, i * 18.0)
        cv2.putText(
            img,
            str(board.SECTORS[i]),
            (int(lx) - 12, int(ly) + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = parse_args(argv)
    if cfg.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    App(cfg).run()


if __name__ == "__main__":
    main()
