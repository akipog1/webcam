"""Visual offline diagnostics for recorded dart-scoring clips.

Creates an annotated MP4 and a per-frame CSV without changing scoring logic.
The displayed quality value is a heuristic shape-quality indicator, NOT a
statistically calibrated probability that a score is correct.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibrator
from .config import DetectorConfig, parse_bool
from .detector import DartDetector
from .scorer import score_point
from .sources import VideoFile


def _shape_metrics(contour: np.ndarray) -> tuple[float, float, float]:
    area = float(cv2.contourArea(contour))
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    aspect = max(w, h) / max(1.0, min(w, h))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area else 0.0
    return area, aspect, solidity


def _quality(area: float, aspect: float, solidity: float, cfg: DetectorConfig) -> float:
    """Transparent heuristic score based on configured contour filters."""
    area_mid = (cfg.min_area + cfg.max_area) / 2.0
    area_span = max(1.0, (cfg.max_area - cfg.min_area) / 2.0)
    area_fit = max(0.0, 1.0 - abs(area - area_mid) / area_span)
    aspect_fit = min(1.0, aspect / max(cfg.min_aspect_ratio * 2.0, 1.0))
    solidity_fit = min(1.0, solidity / max(cfg.min_solidity * 2.0, 0.01))
    return round(100.0 * (0.25 * area_fit + 0.45 * aspect_fit + 0.30 * solidity_fit), 1)


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize ``img`` to fit inside w x h keeping its aspect ratio (letterboxed)."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    out = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = cv2.resize(img, (nw, nh))
    return out


def render_dashboard_frame(
    frame: np.ndarray,
    mask: np.ndarray | None,
    cal: Calibrator,
    detector: DartDetector,
    frame_no: int,
    event: dict | None,
    fps: float,
    cfg: DetectorConfig,
    state: str | None = None,
    past_tips: list | None = None,
) -> np.ndarray:
    """Build a 1280x720 two-panel diagnostic frame.

    ``event`` is the detection to show (may be held over several frames);
    ``past_tips`` are camera-pixel tips of earlier darts, drawn as small markers;
    ``state`` overrides the detector state label.
    """
    canvas = np.zeros((720, 1280, 3), np.uint8)
    source = frame.copy()
    cal.draw_overlay(source)
    for px, py in past_tips or []:
        cv2.drawMarker(source, (int(px), int(py)), (0, 160, 255), cv2.MARKER_CROSS, 18, 2)
    if event and event.get("camera"):
        x, y = event["camera"]
        cv2.drawMarker(source, (int(x), int(y)), (0, 255, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.putText(
            source,
            "estimated tip",
            (int(x) + 12, int(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    canvas[70:520, 20:820] = _fit(source, 800, 450)
    cv2.putText(
        canvas,
        "SOURCE + BOARD OVERLAY",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
    )

    if mask is None:
        mask = np.zeros(frame.shape[:2], np.uint8)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    canvas[70:306, 840:1260] = _fit(mask_bgr, 420, 236)
    cv2.putText(
        canvas, "DIFFERENCE MASK", (840, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2
    )

    changed = int(cv2.countNonZero(mask))
    state = state or detector.state.value
    lines = [
        f"Frame: {frame_no}   Time: {frame_no / max(fps, 1):.2f}s",
        f"State: {state}",
        f"Changed pixels: {changed}",
    ]
    if event:
        lines += [
            f"Detection: {event.get('score', '?')}",
            f"Camera tip: {event['camera'][0]:.1f}, {event['camera'][1]:.1f}",
            f"Board: {event['flat'][0]:.1f}, {event['flat'][1]:.1f}",
            f"Contour area: {event['area']:.1f}px2",
            f"Aspect / solidity: {event['aspect']:.2f} / {event['solidity']:.2f}",
            f"Shape quality: {event['quality']:.0f}/100 (heuristic)",
        ]
    else:
        lines += ["Detection: -", "Tip / score: -"]
    lines += ["Quality is a shape heuristic, not a probability."]
    y = 345
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (840, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 230, 120) if line.startswith("Detection:") and event else (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 31
    cv2.putText(
        canvas,
        "DART SCORER - OFFLINE DIAGNOSTICS",
        (20, 690),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        1,
    )
    return canvas


def run_diagnostics(
    video_path: str,
    calibration_path: str,
    output_path: str,
    csv_path: str,
    baseline_frame: int,
    cfg: DetectorConfig,
    show: bool = False,
) -> int:
    vf = VideoFile(video_path)
    if not vf.is_open():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cal = Calibrator(calibration_path)
    if not cal.load(vf.size):
        vf.release()
        raise ValueError(
            f"No valid calibration for {vf.size[0]}x{vf.size[1]} video: {calibration_path}"
        )

    if baseline_frame < 1:
        vf.release()
        raise ValueError(f"--baseline-frame must be >= 1 (got {baseline_frame})")
    total = int(vf.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if 0 < total < baseline_frame:
        vf.release()
        raise ValueError(
            f"--baseline-frame {baseline_frame} is past the end of the clip ({total} frames)"
        )

    detector = DartDetector(cfg, darts_per_turn=3)
    fps = vf.fps or 30.0
    hold_frames = max(1, int(round(fps)))  # keep each detection on screen ~1 s
    held: dict | None = None
    held_left = 0
    past_tips: list = []
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        vf.release()
        raise OSError(f"Cannot write dashboard video: {output_path}")

    count = detections = 0
    baseline_done = False
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fields_csv = [
            "frame",
            "time_seconds",
            "state",
            "changed_pixels",
            "score",
            "camera_x",
            "camera_y",
            "board_x",
            "board_y",
            "area_px2",
            "aspect_ratio",
            "solidity",
            "shape_quality_heuristic",
        ]
        csvw = csv.DictWriter(fh, fieldnames=fields_csv)
        csvw.writeheader()
        while True:
            frame = vf.read()
            if frame is None:
                break
            count += 1
            event = None
            if count == baseline_frame:
                detector.capture_baseline(frame)
                baseline_done = True
            else:
                detection = detector.update(frame)
                if detection is not None:
                    flat = cal.to_board(detection.tip)
                    score = score_point(*flat)
                    area, aspect, solidity = _shape_metrics(detection.contour)
                    quality = _quality(area, aspect, solidity, cfg)
                    event = {
                        "camera": detection.tip,
                        "flat": flat,
                        "score": score.label,
                        "area": area,
                        "aspect": aspect,
                        "solidity": solidity,
                        "quality": quality,
                    }
                    detections += 1
            state_shown = detector.state.value  # before after_scored(), so SCORED is visible
            detector.after_scored()
            if event:
                if held:
                    past_tips.append(held["camera"])
                held, held_left = event, hold_frames
            elif held_left > 0:
                held_left -= 1
            else:
                held = None
            mask = detector.last_mask
            dashboard = render_dashboard_frame(
                frame,
                mask,
                cal,
                detector,
                count,
                held,
                fps,
                cfg,
                state=state_shown,
                past_tips=past_tips,
            )
            writer.write(dashboard)
            changed = int(cv2.countNonZero(mask)) if mask is not None else 0
            row = {
                "frame": count,
                "time_seconds": round(count / fps, 3),
                "state": state_shown,
                "changed_pixels": changed,
            }
            if event:
                row.update(
                    {
                        "score": event["score"],
                        "camera_x": round(event["camera"][0], 2),
                        "camera_y": round(event["camera"][1], 2),
                        "board_x": round(event["flat"][0], 2),
                        "board_y": round(event["flat"][1], 2),
                        "area_px2": round(event["area"], 2),
                        "aspect_ratio": round(event["aspect"], 3),
                        "solidity": round(event["solidity"], 3),
                        "shape_quality_heuristic": event["quality"],
                    }
                )
            csvw.writerow(row)
            if show:
                cv2.imshow("Dart Scorer Diagnostics", dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    vf.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    if not baseline_done:
        print(f"WARNING: clip ended before --baseline-frame {baseline_frame}; nothing was scored")
    print(f"Frames processed: {count}")
    print(f"Detections: {detections}")
    print(f"Dashboard: {output_path}")
    print(f"Frame log: {csv_path}")
    return 0 if baseline_done else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create an annotated visual diagnostics video")
    ap.add_argument("video")
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--output", default="diagnostics_dashboard.mp4")
    ap.add_argument("--csv", default="diagnostics_frames.csv")
    ap.add_argument("--baseline-frame", type=int, default=5)
    ap.add_argument(
        "--show", action="store_true", help="also display the dashboard while processing"
    )
    for f in fields(DetectorConfig):
        ap.add_argument(
            f"--{f.name.replace('_', '-')}",
            type=parse_bool if f.type in ("bool", bool) else type(f.default),
            default=f.default,
        )
    a = ap.parse_args(argv)
    cfg = DetectorConfig(**{f.name: getattr(a, f.name) for f in fields(DetectorConfig)})
    return run_diagnostics(a.video, a.calibration, a.output, a.csv, a.baseline_frame, cfg, a.show)


if __name__ == "__main__":
    raise SystemExit(main())
