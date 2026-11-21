import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np

from .calibration import MultiCamCalibrator
from .config import AppConfig
from .controller import DualCamScoringController
from .detector import DualCameraFusionDetector, FusedDetection
from .game import GameState
from .sources import DualFrameSource


def run_interactive_calibration(
    source_manager: DualFrameSource,
    calibrator: MultiCamCalibrator,
    save_path: Path,
) -> bool:
    """
    Lets the user click the 4 cardinal outer-double intersections
    (middle of the 20, 6, 3 and 11 doubles) on each camera feed.
    """
    ret, frames = source_manager.read_all()
    if not ret:
        print("[ERROR] Could not read frames for calibration.", file=sys.stderr)
        return False

    prompts = [
        "1/4: TOP - outer edge of the double ring, middle of the 20",
        "2/4: RIGHT - outer edge of the double ring, middle of the 6",
        "3/4: BOTTOM - outer edge of the double ring, middle of the 3",
        "4/4: LEFT - outer edge of the double ring, middle of the 11",
    ]

    for cam_id, frame in frames.items():
        clicked: List[Tuple[float, float]] = []
        win_name = f"Calibrate {cam_id} - Click 4 Cardinal Outer-Double Points"

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
                clicked.append((float(x), float(y)))

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, on_mouse)

        while len(clicked) < 4:
            display = frame.copy()
            cv2.putText(
                display,
                f"{cam_id}: Click {prompts[len(clicked)]} (ESC to cancel)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
            for idx, pt in enumerate(clicked):
                cx, cy = int(pt[0]), int(pt[1])
                cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    str(idx + 1),
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(win_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(win_name)
                print(f"[INFO] Calibration cancelled for {cam_id}.")
                return False

        cv2.destroyWindow(win_name)
        calibrator.calibrate_camera(cam_id, clicked)
        print(f"[INFO] Calibrated {cam_id} with points: {clicked}")

    calibrator.save(save_path)
    print(f"[INFO] Saved dual-camera calibration to {save_path}")
    return True


def draw_overlays(
    frames: Dict[str, np.ndarray],
    calibrator: MultiCamCalibrator,
    controller: DualCamScoringController,
    game: GameState,
    scale: float = 0.65,
) -> np.ndarray:
    """Renders side-by-side camera feeds with calibration rings, tips, and HUD."""
    annotated = []
    last_fused: FusedDetection | None = controller.last_fused

    # Precompute 36-point outer double ring in canonical board space
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ring_pts_mm = [
        (
            calibrator.board_radius_mm * np.cos(a),
            calibrator.board_radius_mm * np.sin(a),
        )
        for a in angles
    ]

    for cam_id, frame in frames.items():
        vis = frame.copy()

        # Draw projected outer double ring if camera is calibrated
        if cam_id in calibrator.inv_homographies:
            ring_px = np.array(
                [calibrator.board_to_pixel(cam_id, pt) for pt in ring_pts_mm],
                dtype=np.int32,
            )
            cv2.polylines(vis, [ring_px], isClosed=True, color=(255, 200, 0), thickness=2)

            # Draw last fused tip reprojected into this camera
            if last_fused is not None:
                fx, fy = calibrator.board_to_pixel(cam_id, last_fused.board_xy)
                cv2.drawMarker(
                    vis,
                    (fx, fy),
                    color=(0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=18,
                    thickness=2,
                )

                # Draw per-camera raw tip detection if present
                cam_det = last_fused.per_camera.get(cam_id)
                if cam_det is not None:
                    rx, ry = int(cam_det.pixel_tip[0]), int(cam_det.pixel_tip[1])
                    cv2.circle(vis, (rx, ry), 5, (0, 255, 0), -1)

        cv2.putText(
            vis,
            f"{cam_id} | State: {controller.state.name}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        if scale != 1.0:
            vis = cv2.resize(vis, (0, 0), fx=scale, fy=scale)
        annotated.append(vis)

    combined = np.hstack(annotated) if len(annotated) > 1 else annotated[0]

    # Top status banner
    banner_h = 42
    banner = np.zeros((banner_h, combined.shape[1], 3), dtype=np.uint8)
    remaining = game.player.remaining
    mode_str = last_fused.mode if last_fused else "waiting"
    hud_text = (
        f"Score Remaining: {remaining} | Darts in Turn: {controller.darts_in_turn}/3 | "
        f"Last Fusion: {mode_str} | Keys: [C]alibrate  [R]eset Board  [Q]uit"
    )
    cv2.putText(
        banner,
        hud_text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 180),
        2,
    )
    return np.vstack([banner, combined])


def parse_source_arg(val: str) -> Union[int, str]:
    return int(val) if val.isdigit() else val


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-Camera Webcam Dart Scorer")
    parser.add_argument(
        "--cam0",
        type=str,
        default="2",
        help="Camera 0 index or video path (default: 2)",
    )
    parser.add_argument(
        "--cam1",
        type=str,
        default="3",
        help="Camera 1 index or video path (default: 3)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Path to calibration.json",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run interactive 4-point calibration on startup",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without OpenCV GUI window",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config = AppConfig(
        camera_sources={
            "cam_0": parse_source_arg(args.cam0),
            "cam_1": parse_source_arg(args.cam1),
        },
        calibration_path=args.calibration,
        show_preview=not args.headless,
    )

    print(f"[INFO] Opening camera streams: {config.camera_sources}")
    sources = DualFrameSource(
        sources=config.camera_sources,
        width=config.frame_width,
        height=config.frame_height,
        fps=config.fps,
    )

    if config.calibration_path.exists():
        calibrator = MultiCamCalibrator.load(config.calibration_path)
        print(f"[INFO] Loaded calibration from {config.calibration_path}")
        missing = [c for c in config.camera_sources if c not in calibrator.homographies]
        if missing:
            print(
                f"[WARN] {config.calibration_path} has no calibration for {missing} "
                "(old single-camera file?). Recalibrating both cameras."
            )
            calibrator = MultiCamCalibrator(board_radius_mm=config.board_radius_mm)
            args.calibrate = True
    else:
        calibrator = MultiCamCalibrator(board_radius_mm=config.board_radius_mm)
        print("[WARN] No calibration.json found; launching interactive calibration.")
        args.calibrate = True

    if args.calibrate:
        run_interactive_calibration(sources, calibrator, config.calibration_path)

    still_missing = [c for c in config.camera_sources if c not in calibrator.homographies]
    if still_missing:
        print(
            f"[ERROR] Cameras not calibrated: {still_missing}. Run again and click all 4 points "
            "on each camera window.",
            file=sys.stderr,
        )
        sources.release()
        cv2.destroyAllWindows()
        sys.exit(1)

    detector = DualCameraFusionDetector(
        cam_ids=("cam_0", "cam_1"),
        max_agreement_mm=config.max_agreement_dist_mm,
    )
    controller = DualCamScoringController(
        calibrator=calibrator,
        detector=detector,
        motion_threshold=config.motion_threshold,
        settle_frames_required=config.settle_frames_required,
        clear_board_threshold=config.clear_board_threshold,
    )
    game = GameState.new(["Player 1"], start=config.starting_score)

    try:
        while True:
            ret, frames = sources.read_all()
            if not ret:
                print("[INFO] End of stream or camera disconnected.")
                break

            event = controller.process_frames(frames)
            if event is not None:
                if event["event"] == "dart_scored":
                    score = event["score"]
                    result = game.add_dart(score)
                    print(
                        f"[DART {event['dart_number']}] {score.label} ({score.points}) | "
                        f"XY: ({event['board_xy'][0]:.1f}, {event['board_xy'][1]:.1f}) mm | "
                        f"Mode: {event['fusion_mode']} (conf={event['confidence']}) | "
                        f"Remaining: {game.player.remaining}"
                        + (f" | {result.message}" if result.message else "")
                    )
                elif event["event"] == "board_cleared":
                    game.next_turn()
                    print(
                        f"[INFO] Board cleared - next turn ({game.player.name}, {game.player.remaining} left)."
                    )

            if config.show_preview:
                canvas = draw_overlays(
                    frames,
                    calibrator,
                    controller,
                    game,
                    scale=config.preview_scale,
                )
                cv2.imshow("Dual-Camera Dart Scorer (Cam 2 & Cam 3)", canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    controller.initialize(frames)
                    print("[INFO] Empty board baseline manually reset.")
                elif key == ord("c"):
                    run_interactive_calibration(sources, calibrator, config.calibration_path)
                    controller.initialize(frames)

    finally:
        sources.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
