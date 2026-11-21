"""Frame sources (live camera, video file) and an optional recorder."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from .config import AppConfig

log = logging.getLogger(__name__)


def _aspects_match(w1: int, h1: int, w2: int, h2: int, tol: float = 0.02) -> bool:
    """True when the two sizes share an aspect ratio within ``tol`` (relative)."""
    if min(w1, h1, w2, h2) <= 0:
        return False
    a1, a2 = w1 / h1, w2 / h2
    return abs(a1 - a2) <= tol * max(a1, a2)


def normalize_camera_frame(
    frame: np.ndarray, requested: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """Fit a captured frame to the requested size without changing aspect.

    Same-aspect frames are resized to ``requested``. A different aspect is
    returned at the camera's native size so calibration is recorded in those pixels.
    """
    fw, fh = int(frame.shape[1]), int(frame.shape[0])
    if _aspects_match(fw, fh, requested[0], requested[1]):
        if (fw, fh) != requested:
            frame = cv2.resize(frame, requested)
        return frame, requested
    return frame, (fw, fh)


class FrameSource(Protocol):
    size: tuple[int, int]  # (width, height) of frames returned by read()
    fps: float
    is_live: bool

    def is_open(self) -> bool: ...
    def read(self) -> Optional[np.ndarray]: ...
    def release(self) -> None: ...


class Camera:
    """VideoCapture wrapper with explicit resolution and reconnects."""

    is_live = True

    def __init__(self, cfg: AppConfig, index: Optional[int] = None) -> None:
        self.cfg = cfg
        self.index = cfg.camera if index is None else index
        self.cap: Optional[cv2.VideoCapture] = None
        self.size = (cfg.width, cfg.height)
        self.fps = 30.0
        self._prefetched: Optional[np.ndarray] = None
        self._aspect_warned = False
        self.open()

    def open(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1:
            self.fps = float(fps)
        self._prefetched = None
        if not self.cap.isOpened():
            return False
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._prefetched = self._normalize(frame)
            return True
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        requested = (self.cfg.width, self.cfg.height)
        if aw > 0 and ah > 0 and not _aspects_match(aw, ah, requested[0], requested[1]):
            self._note_native_size((aw, ah))
        return True

    def _note_native_size(self, size: tuple[int, int]) -> None:
        self.size = size
        if self._aspect_warned:
            return
        self._aspect_warned = True
        log.warning(
            "camera %d delivered %dx%d, aspect differs from requested %dx%d; using native size",
            self.index,
            size[0],
            size[1],
            self.cfg.width,
            self.cfg.height,
        )

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        requested = (self.cfg.width, self.cfg.height)
        out, size = normalize_camera_frame(frame, requested)
        if size != requested:
            self._note_native_size(size)
        else:
            self.size = size
        return out

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        for attempt in range(self.cfg.max_reconnect_attempts):
            if self._prefetched is not None:
                frame = self._prefetched
                self._prefetched = None
                return frame
            ok, raw = (False, None)
            if self.cap is not None:
                ok, raw = self.cap.read()
            if ok and raw is not None:
                return self._normalize(raw)
            log.warning(
                "camera read failed (attempt %d/%d); reconnecting",
                attempt + 1,
                self.cfg.max_reconnect_attempts,
            )
            time.sleep(self.cfg.reconnect_delay)
            if not self.open():
                log.error("could not reopen camera %d", self.index)
        return None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class VideoFile:
    """Replays a recorded clip at its native resolution. ``read`` returns None at the end."""

    is_live = False

    def __init__(self, path: str) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (w, h)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 1 else 30.0

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok and frame is not None else None

    def rewind(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self.cap.release()


class Recorder:
    """Writes the raw frames the detector sees, so a session can be replayed offline."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]) -> None:
        codec = "MJPG" if Path(path).suffix.lower() == ".avi" else "mp4v"
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        self.size = size
        if not self.writer.isOpened():
            raise OSError(f"cannot open {path!r} for recording")
        log.info("recording to %s (%dx%d @ %.0f fps)", path, size[0], size[1], fps)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()


def open_source(cfg: AppConfig) -> FrameSource:
    return VideoFile(cfg.video) if cfg.video else Camera(cfg)


def open_source2(cfg: AppConfig) -> FrameSource:
    """The second camera/clip (only call when ``cfg.dual``)."""
    return VideoFile(cfg.video2) if cfg.video2 else Camera(cfg, cfg.camera2)
