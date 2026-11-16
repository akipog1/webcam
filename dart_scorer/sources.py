import threading
import time
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


class ThreadedCameraStream:
    """Continuously grabs frames in a daemon thread to avoid buffer lag."""

    def __init__(
        self,
        source: Union[int, str],
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        self.is_video_file = isinstance(source, str)
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._stopped = False

        # Grab initial frame synchronously
        self._ret, self._frame = self.cap.read()

        if not self.is_video_file:
            self._thread = threading.Thread(target=self._update, daemon=True)
            self._thread.start()

    def _update(self) -> None:
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                if ret:
                    self._frame = frame
            if not ret:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.is_video_file:
            return self.cap.read()
        with self._lock:
            if not self._ret or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        self._stopped = True
        if not self.is_video_file and hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        self.cap.release()


class DualFrameSource:
    """Manages one or two synchronized camera streams."""

    def __init__(
        self,
        sources: Dict[str, Union[int, str]],
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.streams: Dict[str, ThreadedCameraStream] = {
            cam_id: ThreadedCameraStream(src, width=width, height=height, fps=fps)
            for cam_id, src in sources.items()
        }

    def read_all(self) -> Tuple[bool, Dict[str, np.ndarray]]:
        frames: Dict[str, np.ndarray] = {}
        for cam_id, stream in self.streams.items():
            ret, frame = stream.read()
            if not ret or frame is None:
                return False, {}
            frames[cam_id] = frame
        return True, frames

    def release(self) -> None:
        for stream in self.streams.values():
            stream.release()