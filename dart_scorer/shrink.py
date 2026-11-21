"""Make a recorded clip small enough to share (e.g. under an upload limit) and keep its
calibration valid.

    python -m dart_scorer.shrink session.mp4 --calibration calibration.json --max-mb 25
    python -m dart_scorer.shrink session_cam2.mp4 --calibration calibration2.json --scale 0.6

A calibration is tied to the clip's pixel size, so the tool writes a rescaled copy next to the
shrunk clip (``<name>_calibration.json``). For a two-camera recording shrink both clips with the
same ``--scale`` so the cameras stay in step.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def scale_calibration(
    src: str, dst: str, old_size: tuple[int, int], new_size: tuple[int, int]
) -> None:
    """Write ``dst``: the calibration in ``src`` re-expressed for frames resized old_size -> new_size."""
    data = json.loads(Path(src).read_text())
    s = np.diag([new_size[0] / old_size[0], new_size[1] / old_size[1], 1.0])
    data["H"] = (
        np.array(data["H"], np.float64) @ np.linalg.inv(s)
    ).tolist()  # new px -> old px -> canvas
    data["frame"] = [int(new_size[0]), int(new_size[1])]
    Path(dst).write_text(json.dumps(data, indent=2))


def _write(
    src: str, dst: str, scale: float, every: int, start_s: float, end_s: Optional[float]
) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {src}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nw, nh = max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps / every, (nw, nh))
    if not out.isOpened():
        cap.release()
        raise OSError(f"cannot write {dst}")
    first = int(start_s * fps)
    last = None if end_s is None else int(end_s * fps)
    i = kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= first and (last is None or i < last) and (i - first) % every == 0:
            out.write(
                cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
            )
            kept += 1
        i += 1
    cap.release()
    out.release()
    return w, h, kept


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Shrink a recorded clip and rescale its calibration")
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--calibration", default="", help="calibration.json to rescale alongside the clip"
    )
    ap.add_argument(
        "--max-mb", type=float, default=25.0, help="target size; the scale is lowered until it fits"
    )
    ap.add_argument(
        "--scale", type=float, default=0.0, help="fixed scale (0.1-1.0) instead of --max-mb"
    )
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame (halves fps at 2)")
    ap.add_argument("--start", type=float, default=0.0, help="trim: start second")
    ap.add_argument("--end", type=float, default=None, help="trim: end second")
    a = ap.parse_args(argv)

    src = Path(a.video)
    dst = Path(a.out) if a.out else src.with_name(src.stem + "_small.mp4")
    limit = a.max_mb * 1024 * 1024
    scale = a.scale if a.scale > 0 else 1.0
    while True:
        w, h, kept = _write(str(src), str(dst), scale, max(1, a.every), a.start, a.end)
        size = os.path.getsize(dst)
        if a.scale > 0 or size <= limit or scale <= 0.25:
            break
        scale = max(0.25, scale * min(0.85, (limit / size) ** 0.5))
    nw, nh = max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)
    print(f"{dst}: {size / 1024 / 1024:.1f} MB, {nw}x{nh}, {kept} frames (scale {scale:.2f})")
    if size > limit and a.scale <= 0:
        print(
            f"WARNING: still above {a.max_mb:g} MB at the minimum scale; trim with --start/--end or use --every 2"
        )
    if a.calibration:
        cal_out = dst.with_name(dst.stem + "_calibration.json")
        scale_calibration(a.calibration, str(cal_out), (w, h), (nw, nh))
        print(f"{cal_out}: calibration rescaled for {nw}x{nh}")
    if a.every > 1:
        print(
            f"Frame rate is now 1/{a.every}: use a smaller --settle-frames when replaying (e.g. {max(3, 15 // a.every)})."
        )
    return 0 if size <= limit or a.scale > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
