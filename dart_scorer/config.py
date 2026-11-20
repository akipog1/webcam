"""All tunable parameters in one place, overridable from the CLI."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields


@dataclass
class DetectorConfig:
    """Image-processing thresholds. See README "Tuning guide"."""
    diff_threshold: int = 30          # grey-level change counted as "different"
    blur_kernel: int = 5              # odd; Gaussian blur before differencing
    morph_kernel: int = 3             # opening (removes speckle) + closing
    min_area: float = 40.0            # px^2; smaller blobs = noise
    max_area: float = 6000.0          # px^2; larger = not a single dart
    min_aspect_ratio: float = 2.0     # long side / short side of minAreaRect
    min_solidity: float = 0.2         # area / convex hull area
    settle_frames: int = 15           # ~0.5 s at 30 fps of stable diff before scoring
    stable_pixel_tolerance: float = 0.15  # fraction of mask allowed to change while settling
    hand_motion_area: float = 25000.0 # changed px above this = hand/body, ignore
    background_alpha: float = 0.01    # running-average drift correction while idle
    shadow_v_only: bool = True        # use HSV V channel (shadows mostly change V less sharply)
    tip_method: str = "fitline"       # fitline | lowest | centroid
    camera_side: str = "bottom"       # where the camera axis meets the board edge: bottom|top|left|right


@dataclass
class AppConfig:
    """Runtime options."""
    camera: int = 0
    width: int = 1280
    height: int = 720
    calibration_file: str = "calibration.json"
    log_file: str = "throws.csv"
    game: int = 501
    double_out: bool = True
    players: list[str] = field(default_factory=lambda: ["Player 1", "Player 2"])
    debug: bool = False
    recalibrate: bool = False
    video: str = ""                # replay this clip instead of a live camera
    record: str = ""               # write the raw frames to this file (.mp4/.avi)
    layout: bool = True            # tile the windows automatically
    screen: str = ""               # override screen size, e.g. 1920x1080
    auto_calibrate: bool = True    # try to detect the board before asking for clicks
    # --- optional second camera (enables two-camera fusion) ---
    camera2: int = -1              # >= 0 enables the 2nd camera
    video2: str = ""               # replay a 2nd clip (enables the 2nd camera)
    calibration_file2: str = "calibration2.json"
    record2: str = ""              # defaults to <record>_cam2.<ext> when recording in 2-camera mode
    camera_side2: str = "bottom"   # camera_side for the 2nd camera's tip estimate
    fuse_window: int = 45          # frames to wait for the other camera's detection
    min_cross_angle: float = 20.0  # degrees; below this the two dart lines count as parallel
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 0.5
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    @property
    def dual(self) -> bool:
        return self.camera2 >= 0 or bool(self.video2)


def parse_bool(value: str | bool) -> bool:
    """Parse a boolean command-line value explicitly."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> AppConfig:
    """Build an AppConfig from command-line arguments."""
    p = argparse.ArgumentParser(description="Webcam dart scorer")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--calibration-file", default="calibration.json")
    p.add_argument("--log-file", default="throws.csv")
    p.add_argument("--game", type=int, choices=(301, 501, 701), default=501)
    p.add_argument("--no-double-out", action="store_true")
    p.add_argument("--players", nargs="+", default=["Player 1", "Player 2"])
    p.add_argument("--debug", action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--video", default="", help="replay a recorded clip instead of the camera")
    p.add_argument("--record", default="", help="record the raw camera frames to a file")
    p.add_argument("--no-layout", action="store_true", help="do not tile the windows automatically")
    p.add_argument("--screen", default="", help="screen size for the window layout, e.g. 1920x1080")
    p.add_argument("--no-auto-calibrate", action="store_true", help="skip automatic board detection")
    p.add_argument("--camera2", type=int, default=-1, help="index of a 2nd camera (enables fusion)")
    p.add_argument("--video2", default="", help="2nd recorded clip (replay in 2-camera mode)")
    p.add_argument("--calibration-file2", default="calibration2.json")
    p.add_argument("--record2", default="")
    p.add_argument("--camera-side2", choices=("bottom", "top", "left", "right"), default="bottom")
    p.add_argument("--fuse-window", type=int, default=45)
    p.add_argument("--min-cross-angle", type=float, default=20.0)
    for f in fields(DetectorConfig):
        p.add_argument(f"--{f.name.replace('_', '-')}",
                       type=parse_bool if f.type in ("bool", bool) else type(f.default),
                       default=f.default)
    a = p.parse_args(argv)
    det = DetectorConfig(**{f.name: getattr(a, f.name) for f in fields(DetectorConfig)})
    return AppConfig(camera=a.camera, width=a.width, height=a.height,
                     calibration_file=a.calibration_file, log_file=a.log_file,
                     game=a.game, double_out=not a.no_double_out, players=a.players,
                     debug=a.debug, recalibrate=a.recalibrate, video=a.video, record=a.record,
                     layout=not a.no_layout, screen=a.screen, auto_calibrate=not a.no_auto_calibrate,
                     camera2=a.camera2, video2=a.video2, calibration_file2=a.calibration_file2,
                     record2=a.record2, camera_side2=a.camera_side2, fuse_window=a.fuse_window,
                     min_cross_angle=a.min_cross_angle, detector=det)
