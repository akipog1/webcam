from enum import Enum, auto
from typing import Dict, Optional
import cv2
import numpy as np
from .calibration import MultiCamCalibrator
from .detector import DualCameraFusionDetector, FusedDetection
from .scorer import score_board_mm


class TurnState(Enum):
    IDLE = auto()
    MOTION = auto()
    SETTLING = auto()
    PULLING_DARTS = auto()


class DualCamScoringController:
    """State machine synchronizing motion detection, settling, and fusion scoring."""

    def __init__(
        self,
        calibrator: MultiCamCalibrator,
        detector: DualCameraFusionDetector,
        motion_threshold: float = 0.004,
        settle_frames_required: int = 6,
        clear_board_threshold: float = 0.03,
    ):
        self.calibrator = calibrator
        self.detector = detector
        self.motion_threshold = motion_threshold
        self.settle_frames_required = settle_frames_required
        self.clear_board_threshold = clear_board_threshold

        self.state = TurnState.IDLE
        self.prev_frames: Dict[str, np.ndarray] = {}
        self.empty_board_frames: Dict[str, np.ndarray] = {}
        self.settle_counter = 0
        self.darts_in_turn = 0
        self.last_fused: Optional[FusedDetection] = None

    def initialize(self, frames: Dict[str, np.ndarray]) -> None:
        self.prev_frames = {k: v.copy() for k, v in frames.items()}
        self.empty_board_frames = {k: v.copy() for k, v in frames.items()}
        self.detector.reset_all(frames)
        self.state = TurnState.IDLE
        self.darts_in_turn = 0

    def process_frames(self, frames: Dict[str, np.ndarray]) -> Optional[dict]:
        """
        Processes synchronized frames from all active cameras.
        Returns a score event dictionary when a new dart throw settles, else None.
        """
        if not self.prev_frames:
            self.initialize(frames)
            return None

        # Compute max motion across both cameras
        max_motion = 0.0
        for cid, frame in frames.items():
            if cid in self.prev_frames and cid in self.detector.detectors:
                m = self.detector.detectors[cid].motion_score(self.prev_frames[cid], frame)
                max_motion = max(max_motion, m)

        self.prev_frames = {k: v.copy() for k, v in frames.items()}

        # Detect if player is pulling darts (large sustained motion after 3 darts or manual pull)
        if max_motion > self.clear_board_threshold:
            self.state = TurnState.MOTION
            self.settle_counter = 0
            return None

        if self.state == TurnState.IDLE:
            if max_motion > self.motion_threshold:
                self.state = TurnState.MOTION
                self.settle_counter = 0
            return None

        if self.state == TurnState.MOTION:
            if max_motion <= self.motion_threshold:
                self.state = TurnState.SETTLING
                self.settle_counter = 1
            return None

        if self.state == TurnState.SETTLING:
            if max_motion > self.motion_threshold:
                # Dart is still wobbling or a second dart was thrown immediately
                self.state = TurnState.MOTION
                self.settle_counter = 0
                return None

            self.settle_counter += 1
            if self.settle_counter < self.settle_frames_required:
                return None

            # Both cameras are settled -> run detection & fusion
            self.state = TurnState.IDLE
            self.settle_counter = 0

            # Check if board was cleared back to empty baseline
            if self._is_board_cleared(frames):
                self.detector.reset_all(frames)
                self.darts_in_turn = 0
                return {"event": "board_cleared"}

            fused = self.detector.detect_and_fuse(frames, self.calibrator)
            if fused is None:
                return None

            self.last_fused = fused
            self.darts_in_turn += 1
            self.detector.commit_throw(frames, fused)

            score_result = score_board_mm(*fused.board_xy)
            return {
                "event": "dart_scored",
                "dart_number": self.darts_in_turn,
                "board_xy": fused.board_xy,
                "fusion_mode": fused.mode,
                "confidence": round(fused.confidence, 3),
                "score": score_result,
            }

        return None

    def _is_board_cleared(self, frames: Dict[str, np.ndarray]) -> bool:
        """Checks if current frames match the original empty board baseline."""
        if self.darts_in_turn == 0 or not self.empty_board_frames:
            return False

        diffs = []
        for cid, frame in frames.items():
            if cid in self.empty_board_frames:
                base_gray = cv2.cvtColor(self.empty_board_frames[cid], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                delta = cv2.absdiff(base_gray, curr_gray)
                _, thresh = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)
                diffs.append(np.count_nonzero(thresh) / thresh.size)

        return bool(diffs) and max(diffs) < 0.005