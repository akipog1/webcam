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
        return self.cap.isOpened()

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        for attempt in range(self.cfg.max_reconnect_attempts):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if (frame.shape[1], frame.shape[0]) != self.size:
                    frame = cv2.resize(frame, self.size)
                return frame
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
