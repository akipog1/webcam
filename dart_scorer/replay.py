"""Headless replay harness: run a recorded clip through detector + scorer and
compare against the scores you actually threw.

    python -m dart_scorer.replay clip.mp4 --expect expected.txt \\
        [--calibration calibration.json] [--baseline-frame 5] [--min-accuracy 0.9] \\
        [--diff-threshold 25 --settle-frames 20 ...]

``expected.txt``: scores in throw order, whitespace or comma separated, ``#``
comments allowed (``T20 D16 S5 50 25 0``). Compare by parsed score, so ``5`` == ``S5``.
The clip must start with an empty board (the baseline is taken at
``--baseline-frame``); the detector re-arms itself when darts are pulled.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from itertools import zip_longest
from typing import Iterable, Optional, Sequence

import numpy as np

from .calibration import Calibrator
from .config import DetectorConfig, parse_bool
from .detector import DartDetector
from .fusion import DetectionPairer, fuse
from .scorer import Score, parse_score, score_point
from .sources import VideoFile


@dataclass
class Row:
    index: int
    expected: Optional[Score]
    got: Optional[Score]
    frame: Optional[int]
    flat: Optional[tuple[float, float]]

    @property
    def ok(self) -> bool:
        return (self.expected is not None and self.got is not None
                and (self.expected.label, self.expected.points) == (self.got.label, self.got.points))


@dataclass
class ReplayResult:
    rows: list[Row]
    frames: int

    @property
    def accuracy(self) -> float:
        exp = [r for r in self.rows if r.expected is not None]
        return sum(r.ok for r in exp) / len(exp) if exp else 1.0

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.rows)


def parse_expected(text: str) -> list[Score]:
    out: list[Score] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        out += [parse_score(tok) for tok in line.replace(",", " ").split()]
    return out


def replay(frames: Iterable[np.ndarray], cal: Calibrator, cfg: DetectorConfig,
           expected: Sequence[Score] = (), baseline_frame: int = 5) -> ReplayResult:
    """Run frames through the pipeline. Detections are matched to ``expected`` by order."""
    if baseline_frame < 1:
        raise ValueError(f"--baseline-frame must be >= 1 (got {baseline_frame})")
    det = DartDetector(cfg, darts_per_turn=3)
    got: list[tuple[Score, int, tuple[float, float]]] = []
    n = 0
    for n, frame in enumerate(frames, start=1):
        if n == baseline_frame:
            det.capture_baseline(frame)
            continue
        d = det.update(frame)
        det.after_scored()
        if d is not None:
            flat = cal.to_board(d.tip)
            got.append((score_point(*flat), n, flat))
    if n < baseline_frame:
        raise ValueError(f"--baseline-frame {baseline_frame} is past the end of the clip ({n} frames)")
    rows = []
    for i, (e, g) in enumerate(zip_longest(expected, got), start=1):
        rows.append(Row(i, e, g[0] if g else None, g[1] if g else None, g[2] if g else None))
    return ReplayResult(rows, n)


def replay_dual(frames1: Iterable[np.ndarray], frames2: Iterable[np.ndarray],
                cal1: Calibrator, cal2: Calibrator, cfg: DetectorConfig,
                expected: Sequence[Score] = (), baseline_frame: int = 5,
                window: int = 45, min_angle: float = 20.0, cfg2: Optional[DetectorConfig] = None
                ) -> ReplayResult:
    """Two-camera replay: both clips are stepped in lock-step and detections fused."""
    if baseline_frame < 1:
        raise ValueError(f"--baseline-frame must be >= 1 (got {baseline_frame})")
    d1, d2 = DartDetector(cfg, darts_per_turn=3), DartDetector(cfg2 or cfg, darts_per_turn=3)
    pairer = DetectionPairer(window)
    got: list[tuple[Score, int, tuple[float, float]]] = []
    n = 0

    def emit(pairs):
        for a, b in pairs:
            tip = fuse(a, cal1, b, cal2, min_angle)
            if tip is not None:
                got.append((score_point(*tip.flat), n, tip.flat))

    for n, (f1, f2) in enumerate(zip(frames1, frames2), start=1):
        if n == baseline_frame:
            d1.capture_baseline(f1)
            d2.capture_baseline(f2)
            continue
        ready = []
        for cam, det, frame in ((1, d1, f1), (2, d2, f2)):
            d = det.update(frame)
            det.after_scored()
            if d is not None:
                ready += pairer.push(cam, d)
        ready += pairer.tick()
        emit(ready)
    if n < baseline_frame:
        raise ValueError(f"--baseline-frame {baseline_frame} is past the end of the clips ({n} frames)")
    for cam, (_, det) in list(pairer._pending.items()):   # clip ended with a lone detection
        emit([(det, None) if cam == 1 else (None, det)])
    rows = []
    for i, (e, g) in enumerate(zip_longest(expected, got), start=1):
        rows.append(Row(i, e, g[0] if g else None, g[1] if g else None, g[2] if g else None))
    return ReplayResult(rows, n)


def format_table(res: ReplayResult) -> str:
    lines = [f"{'#':>3}  {'expected':<10} {'got':<10} {'frame':>6}  {'flat (x, y)':<14} result"]
    for r in res.rows:
        e = r.expected.label if r.expected else "-"
        g = r.got.label if r.got else "(missed)"
        pos = f"{r.flat[0]:.0f}, {r.flat[1]:.0f}" if r.flat else ""
        verdict = "ok" if r.ok else ("EXTRA" if r.expected is None else "MISS" if r.got is None else "WRONG")
        lines.append(f"{r.index:>3}  {e:<10} {g:<10} {r.frame or '':>6}  {pos:<14} {verdict}")
    n = sum(r.expected is not None for r in res.rows)
    hit = sum(r.ok for r in res.rows)
    lines.append(f"\n{hit}/{n} correct ({res.accuracy:.0%}), {res.frames} frames")
    return "\n".join(lines)


def _video_frames(vf: VideoFile):
    while (f := vf.read()) is not None:
        yield f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a clip and compare with expected scores")
    ap.add_argument("video")
    ap.add_argument("--expect", help="file with the scores in throw order")
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--video2", help="second camera's clip: enables two-camera fusion")
    ap.add_argument("--calibration2", default="calibration2.json")
    ap.add_argument("--camera-side2", choices=("bottom", "top", "left", "right"), default="bottom")
    ap.add_argument("--fuse-window", type=int, default=45)
    ap.add_argument("--min-cross-angle", type=float, default=20.0)
    ap.add_argument("--baseline-frame", type=int, default=5)
    ap.add_argument("--min-accuracy", type=float, default=1.0,
                    help="exit non-zero below this fraction correct (default 1.0)")
    for f in fields(DetectorConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}",
                        type=parse_bool if f.type in ("bool", bool) else type(f.default),
                        default=f.default)
    a = ap.parse_args(argv)
    cfg = DetectorConfig(**{f.name: getattr(a, f.name) for f in fields(DetectorConfig)})

    vf = VideoFile(a.video)
    if not vf.is_open():
        print(f"cannot open {a.video}", file=sys.stderr)
        return 2
    cal = Calibrator(a.calibration)
    if not cal.load(vf.size):
        print(f"no usable calibration at {a.calibration} for {vf.size[0]}x{vf.size[1]} video", file=sys.stderr)
        return 2
    expected = []
    if a.expect:
        with open(a.expect) as fh:
            expected = parse_expected(fh.read())
    vf2 = None
    try:
        if a.video2:
            vf2 = VideoFile(a.video2)
            cal2 = Calibrator(a.calibration2)
            if not vf2.is_open():
                print(f"cannot open {a.video2}", file=sys.stderr)
                vf.release()
                return 2
            if not cal2.load(vf2.size):
                print(f"no usable calibration at {a.calibration2} for {vf2.size[0]}x{vf2.size[1]} video", file=sys.stderr)
                vf.release()
                vf2.release()
                return 2
            import dataclasses
            res = replay_dual(_video_frames(vf), _video_frames(vf2), cal, cal2, cfg, expected,
                              a.baseline_frame, a.fuse_window, a.min_cross_angle,
                              dataclasses.replace(cfg, camera_side=a.camera_side2))
            vf2.release()
        else:
            res = replay(_video_frames(vf), cal, cfg, expected, a.baseline_frame)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        vf.release()
        if vf2 is not None:
            vf2.release()
        return 2
    vf.release()
    print(format_table(res))
    if not a.expect:
        return 0
    return 0 if res.accuracy >= a.min_accuracy and all(r.expected is not None for r in res.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
