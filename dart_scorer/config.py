from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union


@dataclass
class AppConfig:
    """Configuration for dual-camera dart scoring."""

    # Default USB camera indices 2 & 3 (can also be overridden with MP4 paths)
    camera_sources: Dict[str, Union[int, str]] = field(
        default_factory=lambda: {
            "cam_0": 2,
            "cam_1": 3,
        }
    )

    # Capture settings (MJPG at 720p keeps dual USB cameras smooth on one bus)
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30

    # Calibration file path
    calibration_path: Path = Path("calibration.json")

    # Board geometry & fusion settings
    board_radius_mm: float = 170.0
    max_agreement_dist_mm: float = 12.0

    # Motion & settle state machine parameters
    motion_threshold: float = 0.004
    settle_frames_required: int = 6
    clear_board_threshold: float = 0.03

    # Game & UI settings
    starting_score: int = 501
    show_preview: bool = True
    preview_scale: float = 0.65