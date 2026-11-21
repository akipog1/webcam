"""Tile the app's windows so everything is visible at once (no manual dragging).

``compute_layout`` is pure (testable); ``apply_layout`` talks to OpenCV and never raises.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2

log = logging.getLogger(__name__)

TITLE_BAR = 32  # window title bar height (px), added to every window's client height
TASKBAR = 56  # space kept free at the bottom of the screen
DEFAULT_SCREEN = (1920, 1080)
Rect = tuple[int, int, int, int]  # x, y, client width, client height


def screen_size(override: str = "") -> tuple[int, int]:
    """Primary screen size: ``--screen WxH`` override, else Windows API, else tkinter, else 1920x1080."""
    if override:
        try:
            w, h = (int(v) for v in override.lower().split("x"))
            return w, h
        except ValueError:
            log.warning("bad --screen %r (use e.g. 1920x1080)", override)
    try:
        import ctypes

        u = ctypes.windll.user32  # type: ignore[attr-defined]
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        pass
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return DEFAULT_SCREEN


def compute_layout(
    screen: tuple[int, int],
    dual: bool,
    cam_size: tuple[int, int] = (1280, 720),
    cam2_size: Optional[tuple[int, int]] = None,
    debug: bool = False,
) -> dict[str, Rect]:
    """Window name -> (x, y, w, h).

    Single camera: the camera view on the left, the flat Board square on the right.
    Two cameras: both camera views stacked on the left, the Board square on the right.
    """
    sw, sh = screen
    avail_h = max(300, sh - TASKBAR)
    rows = 2 if dual else 1
    aspect = cam_size[0] / cam_size[1]
    h = (avail_h - rows * TITLE_BAR) // rows
    w = int(h * aspect)
    max_w = int(sw * (0.6 if dual else 0.65))
    if w > max_w:
        w, h = max_w, int(max_w / aspect)
    out: dict[str, Rect] = {"Dart Scorer": (0, 0, w, h)}
    if dual:
        a2 = (cam2_size[0] / cam2_size[1]) if cam2_size else aspect
        h2 = int(w / a2)
        out["Dart Scorer 2"] = (0, h + TITLE_BAR, w, h2)
    side = max(200, min(sw - w - 8, avail_h - TITLE_BAR))
    if debug:  # leave room under the Board for the Mask window
        side = min(side, int(avail_h * 0.62) - TITLE_BAR)
    out["Board"] = (w + 8, 0, side, side)
    if debug:
        mh = avail_h - (side + TITLE_BAR) - TITLE_BAR
        mw = min(int(mh * aspect), sw - w - 8)
        out["Mask"] = (w + 8, side + TITLE_BAR, mw, int(mw / aspect))
    return out


def image_point_from_window(
    window: str, x: float, y: float, image_size: tuple[int, int]
) -> tuple[float, float]:
    """Map a highgui mouse position into image pixels.

    The Windows backend reports client coordinates. When the window is a
    different size from the image, scale by the image rectangle. If the window
    cannot be queried (headless tests, missing window), return the point as given.
    """
    iw, ih = image_size
    if iw <= 0 or ih <= 0:
        return float(x), float(y)
    try:
        rect = cv2.getWindowImageRect(window)
    except cv2.error:
        return float(x), float(y)
    if rect is None or len(rect) < 4:
        return float(x), float(y)
    ww, wh = int(rect[2]), int(rect[3])
    if ww <= 0 or wh <= 0 or (ww == iw and wh == ih):
        return float(x), float(y)
    return float(x) * iw / ww, float(y) * ih / wh


def apply_layout(layout: dict[str, Rect]) -> None:
    """Create each window as resizable and put it in place."""
    for name, (x, y, w, h) in layout.items():
        try:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, w, h)
            cv2.moveWindow(name, x, y)
        except cv2.error as e:  # headless build, odd backend: the app still works, just untiled
            log.debug("could not place window %s: %s", name, e)
            return
