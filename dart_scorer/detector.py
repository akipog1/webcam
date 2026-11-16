from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from .calibration import MultiCamCalibrator


@dataclass
class CameraDetection:
    cam_id: str
    pixel_tip: Tuple[float, float]
    board_xy: Tuple[float, float]
    confidence: float
    area: float
    elongation: float
    mask: np.ndarray


@dataclass
class FusedDetection:
    board_xy: Tuple[float, float]
    mode: str
    confidence: float
    per_camera: Dict[str, Optional[CameraDetection]]


class SingleCameraDartDetector:
    """Background-difference dart detector with tip extraction and confidence scoring."""

    def __init__(
        self,
        cam_id: str,
        min_contour_area: float = 180.0,
        max_contour_area: float = 25000.0,
        diff_threshold: int = 32,
    ):
        self.cam_id = cam_id
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.diff_threshold = diff_threshold

        self.reference_gray: Optional[np.ndarray] = None
        self.accumulated_dart_mask: Optional[np.ndarray] = None

    def reset_reference(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self.accumulated_dart_mask = np.zeros_like(gray, dtype=np.uint8)

    def update_reference_with_dart(self, frame: np.ndarray, dart_mask: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.accumulated_dart_mask is not None:
            self.accumulated_dart_mask = cv2.bitwise_or(self.accumulated_dart_mask, dart_mask)

    def motion_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(g1, g2)
        _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        return float(np.count_nonzero(thresh)) / float(thresh.size)

    def detect(
        self, frame: np.ndarray, calibrator: MultiCamCalibrator
    ) -> Optional[CameraDetection]:
        if self.reference_gray is None:
            self.reset_reference(frame)
            return None

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = cv2.absdiff(self.reference_gray, gray)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [
            c for c in contours
            if self.min_contour_area <= cv2.contourArea(c) <= self.max_contour_area
        ]
        if not valid:
            return None

        # Merge split fragments (e.g., barrel + flight separated by wire reflection)
        merged_pts = np.vstack(valid)
        area = float(sum(cv2.contourArea(c) for c in valid))

        # Fit minimum area bounding rectangle to measure dart elongation
        rect = cv2.minAreaRect(merged_pts)
        w, h = rect[1]
        short_side = max(min(w, h), 1.0)
        long_side = max(w, h)
        elongation = float(long_side / short_side)

        # Find extreme endpoints along principal axis; the point closer to board center
        # in canonical space corresponds to the steel tip embedded in the sisal
        pts = merged_pts.reshape(-1, 2).astype(np.float32)
        mean = np.mean(pts, axis=0)
        _, eigenvectors = cv2.PCACompute(pts, mean=mean.reshape(1, 2))
        axis = eigenvectors[0]

        projections = (pts - mean) @ axis
        end_a = tuple(pts[np.argmin(projections)])
        end_b = tuple(pts[np.argmax(projections)])

        board_a = calibrator.pixel_to_board(self.cam_id, end_a)
        board_b = calibrator.pixel_to_board(self.cam_id, end_b)

        # The flight sticks outward away from the board plane, projecting further from center
        if np.hypot(*board_a) <= np.hypot(*board_b):
            pixel_tip, board_xy = end_a, board_a
        else:
            pixel_tip, board_xy = end_b, board_b

        # Compute confidence (0.1 to 1.0) based on elongation & occlusion overlap
        elongation_score = min(elongation / 4.5, 1.0)

        dart_mask = np.zeros_like(mask)
        cv2.drawContours(dart_mask, valid, -1, 255, thickness=cv2.FILLED)

        occlusion_penalty = 0.0
        if self.accumulated_dart_mask is not None:
            overlap = cv2.bitwise_and(dart_mask, self.accumulated_dart_mask)
            overlap_ratio = np.count_nonzero(overlap) / max(np.count_nonzero(dart_mask), 1)
            occlusion_penalty = min(overlap_ratio * 0.8, 0.7)

        confidence = float(np.clip(elongation_score * (1.0 - occlusion_penalty), 0.1, 1.0))

        return CameraDetection(
            cam_id=self.cam_id,
            pixel_tip=pixel_tip,
            board_xy=board_xy,
            confidence=confidence,
            area=area,
            elongation=elongation,
            mask=dart_mask,
        )


class DualCameraFusionDetector:
    """Fuses board-plane detections from two calibrated cameras."""

    def __init__(self, cam_ids: Tuple[str, str] = ("cam_0", "cam_1"), max_agreement_mm: float = 12.0):
        self.cam_ids = cam_ids
        self.max_agreement_mm = max_agreement_mm
        self.detectors: Dict[str, SingleCameraDartDetector] = {
            cid: SingleCameraDartDetector(cid) for cid in cam_ids
        }

    def reset_all(self, frames: Dict[str, np.ndarray]) -> None:
        for cid, det in self.detectors.items():
            if cid in frames:
                det.reset_reference(frames[cid])

    def commit_throw(self, frames: Dict[str, np.ndarray], fused: FusedDetection) -> None:
        for cid, det in self.detectors.items():
            cam_det = fused.per_camera.get(cid)
            if cam_det is not None:
                det.update_reference_with_dart(frames[cid], cam_det.mask)
            elif cid in frames:
                det.update_reference_with_dart(frames[cid], np.zeros(frames[cid].shape[:2], dtype=np.uint8))

    def detect_and_fuse(
        self, frames: Dict[str, np.ndarray], calibrator: MultiCamCalibrator
    ) -> Optional[FusedDetection]:
        c0, c1 = self.cam_ids
        det0 = self.detectors[c0].detect(frames[c0], calibrator) if c0 in frames else None
        det1 = self.detectors[c1].detect(frames[c1], calibrator) if c1 in frames else None

        per_cam = {c0: det0, c1: det1}

        if det0 is None and det1 is None:
            return None

        if det0 is not None and det1 is None:
            return FusedDetection(det0.board_xy, f"single_{c0}", det0.confidence * 0.85, per_cam)

        if det1 is not None and det0 is None:
            return FusedDetection(det1.board_xy, f"single_{c1}", det1.confidence * 0.85, per_cam)

        # Both cameras detected a dart
        p0 = np.array(det0.board_xy, dtype=np.float32)
        p1 = np.array(det1.board_xy, dtype=np.float32)
        dist = float(np.linalg.norm(p0 - p1))

        if dist <= self.max_agreement_mm:
            w0, w1 = det0.confidence, det1.confidence
            fused_xy = (p0 * w0 + p1 * w1) / (w0 + w1)
            combined_conf = min(1.0, (w0 + w1) / 1.5)
            return FusedDetection(
                board_xy=(float(fused_xy[0]), float(fused_xy[1])),
                mode="fused_dual",
                confidence=combined_conf,
                per_camera=per_cam,
            )

        # Disagreement: select camera with cleaner elongation / lower occlusion
        if det0.confidence >= det1.confidence:
            return FusedDetection(det0.board_xy, f"conflict_{c0}", det0.confidence, per_cam)
        return FusedDetection(det1.board_xy, f"conflict_{c1}", det1.confidence, per_cam)