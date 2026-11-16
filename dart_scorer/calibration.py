import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np


# Standard WDF/PDC double-wire outer radius in millimeters
OUTER_DOUBLE_RADIUS_MM = 170.0


def canonical_calibration_targets(radius_mm: float = OUTER_DOUBLE_RADIUS_MM) -> np.ndarray:
    """
    Returns the 4 standard cardinal outer-double wire intersections in mm:
    Top (20/1-5 border), Right (6/13-10), Bottom (3/19-17), Left (11/14-8).
    """
    return np.array(
        [
            [0.0, -radius_mm],  # Top (12 o'clock)
            [radius_mm, 0.0],   # Right (3 o'clock)
            [0.0, radius_mm],   # Bottom (6 o'clock)
            [-radius_mm, 0.0],  # Left (9 o'clock)
        ],
        dtype=np.float32,
    )


class MultiCamCalibrator:
    """Stores and applies per-camera 3x3 homography transforms into board space."""

    def __init__(self, board_radius_mm: float = OUTER_DOUBLE_RADIUS_MM):
        self.board_radius_mm = board_radius_mm
        self.homographies: Dict[str, np.ndarray] = {}
        self.inv_homographies: Dict[str, np.ndarray] = {}
        self.src_points: Dict[str, List[List[float]]] = {}

    def calibrate_camera(
        self, cam_id: str, clicked_points: List[Tuple[float, float]]
    ) -> np.ndarray:
        if len(clicked_points) != 4:
            raise ValueError("Exactly 4 calibration points are required.")

        src = np.array(clicked_points, dtype=np.float32)
        dst = canonical_calibration_targets(self.board_radius_mm)

        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise RuntimeError(f"Homography computation failed for {cam_id}")

        self.src_points[cam_id] = [list(pt) for pt in clicked_points]
        self.homographies[cam_id] = H
        self.inv_homographies[cam_id] = np.linalg.inv(H)
        return H

    def pixel_to_board(self, cam_id: str, pixel_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Projects a single (u, v) pixel coordinate into canonical board (X, Y) mm."""
        H = self.homographies[cam_id]
        pt = np.array([[[pixel_xy[0], pixel_xy[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)[0][0]
        return float(transformed[0]), float(transformed[1])

    def board_to_pixel(self, cam_id: str, board_xy: Tuple[float, float]) -> Tuple[int, int]:
        """Projects canonical board (X, Y) mm back into camera pixel space for overlays."""
        H_inv = self.inv_homographies[cam_id]
        pt = np.array([[[board_xy[0], board_xy[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H_inv)[0][0]
        return int(round(transformed[0])), int(round(transformed[1]))

    def save(self, path: Union[str, Path] = "calibration.json") -> None:
        payload = {
            "board_radius_mm": self.board_radius_mm,
            "cameras": {
                cam_id: {
                    "src_points": self.src_points[cam_id],
                    "homography_matrix": self.homographies[cam_id].tolist(),
                }
                for cam_id in self.homographies
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path] = "calibration.json") -> "MultiCamCalibrator":
        data = json.loads(Path(path).read_text())
        instance = cls(board_radius_mm=data.get("board_radius_mm", OUTER_DOUBLE_RADIUS_MM))

        # Support legacy single-camera calibration.json seamlessly
        if "cameras" not in data and "homography_matrix" in data:
            H = np.array(data["homography_matrix"], dtype=np.float32)
            instance.homographies["cam_0"] = H
            instance.inv_homographies["cam_0"] = np.linalg.inv(H)
            instance.src_points["cam_0"] = data.get("src_points", [])
            return instance

        for cam_id, cam_data in data.get("cameras", {}).items():
            H = np.array(cam_data["homography_matrix"], dtype=np.float32)
            instance.homographies[cam_id] = H
            instance.inv_homographies[cam_id] = np.linalg.inv(H)
            instance.src_points[cam_id] = cam_data.get("src_points", [])

        return instance