# Webcam Dart Scorer v2

Real-time steel-tip dart scoring with one webcam, OpenCV and numpy.

## Setup
For development, install the project and its test/lint tools in editable mode:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[test,dev]"
```

For a runtime-only installation from the dependency list:
```bash
python -m pip install -r requirements.txt
```

Then launch the scorer:
```bash
python -m dart_scorer --camera 0 --width 1280 --height 720 --players Alice Bob --game 501
```
To run tests and lint checks in the development environment, use `python -m pytest`,
`python -m ruff check .`, and `python -m ruff format --check .`. Python 3.10+ is required.
Useful flags: `--debug`, `--recalibrate`, `--no-double-out`, `--game 301`,
plus every detector setting, e.g. `--diff-threshold 25 --settle-frames 20 --tip-method lowest --camera-side left`.

## Calibration
Runs automatically on first start (or `--recalibrate`, or key `c`). Click in order:
1. Bull centre  2. D20 outer wire (top)  3. D6 outer wire (right)  4. D3 outer wire (bottom)  5. D11 outer wire (left)

Click the point where the **outer double wire crosses the middle of that number's segment**.
After 4 points the ideal board is drawn over the live video; check that the rings sit on the wires,
then press **Enter**. With 5 points RANSAC ignores one bad click. `u` undo, `r` reset, `q` quit.
The result is saved to `calibration.json` and reused next time.

## Controls (main window)
| Key | Action |
|---|---|
| `b` | capture empty-board baseline |
| `n` | next turn / player (also resets the baseline) |
| `u` | undo last dart |
| `e` | type a score override (`T20`, `D16`, `S5`, `25`, `50`, `0`), Enter to apply, Esc to cancel |
| click in "Board" window | re-score last dart at that point |
| `c` | recalibrate |
| `q` | quit |

After the 3rd dart the detector waits for the darts to be pulled; once the board looks like the
turn-start baseline again it resets automatically. Press `n` to advance players.
Each accepted throw is written to `throws.csv` (timestamp, player, raw px, flat coords, ring, sector, points, label, source). Darts the game refuses (turn already full, after a bust, after a win) are not logged. Undo appends a row with source `undo`, which voids the previous row for that player. Typed `e` scores accept `B` for bull.

## Record and replay
Record a session's raw frames (the detector's exact input) and tune offline instead of throwing darts all evening:
```bash
python -m dart_scorer --record session.mp4          # .avi also works (MJPG)
python -m dart_scorer --video session.mp4           # replay in the full UI; calibration loops the clip
```
Headless accuracy check against the scores you really threw (start the clip with an empty board):
```bash
echo "T20 D16 S5 50 25 0" > expected.txt             # throw order; # comments allowed
python -m dart_scorer.replay session.mp4 --expect expected.txt --calibration calibration.json \
    --diff-threshold 25 --settle-frames 20
```
Prints a per-dart table (`ok / WRONG / MISS / EXTRA`) and the accuracy; exit code 1 below `--min-accuracy`
(default 1.0). The calibration must have been made at the clip's resolution. Detector flags are the same as the live app,
so you can sweep settings against the same clip.

## Two cameras (v2.3)
A second camera removes the depth ambiguity of a single view: each camera's dart axis is mapped onto the
board plane, and the two lines intersect at the true tip.
```bash
python -m dart_scorer --camera 2 --camera2 3 --players Alice Bob --camera-side2 bottom
```
- Each camera is calibrated separately on first start (calibration.json and calibration2.json; `--recalibrate` or key `c` redoes both).
- Windows: **Dart Scorer** and **Board** (camera 1), **Dart Scorer 2** (camera 2).
- **Remove all darts before pressing `b`** (it baselines both cameras); `n` and `u` act on both.
- Best with the cameras 60-90 degrees apart around the board (e.g. one low in front, one at the side).
  Lines closer than `--min-cross-angle` (20 deg) fall back to averaging both tips; if only one camera sees a dart, that one is used.
  The `source` column in `throws.csv` says which: `auto-intersection`, `auto-average`, `auto-cam1`, `auto-cam2`.
- `--fuse-window 45`: frames to wait for the other camera. Detector flags apply to both cameras; `--camera-side2` sets camera 2's tip direction (only used in fallbacks).
- Use separate USB ports/controllers; if frames drop, lower `--width/--height` (recalibrate after changing).
- Record both: `--record s.mp4` also writes `s_cam2.mp4`. Replay: `--video s.mp4 --video2 s_cam2.mp4` (UI) or
  `python -m dart_scorer.replay s.mp4 --video2 s_cam2.mp4 --calibration calibration.json --calibration2 calibration2.json --expect expected.txt`.
- Tests use a simulated 3-D rig (`tests/synth3d.py`): ~0.5 mm fused error vs 3-4 mm single camera. Real boards will be worse; check with your own clips.

## Camera placement
- **Default (`--camera-side bottom`)**: camera below the board, 30–60 cm in front, looking up at ~30–45°.
  Darts lean away from it, so the tip is the end of the dart nearest the bottom of the image.
- **Side-mounted**: use `--camera-side left` or `right`. Best accuracy for a single camera, since
  the tip is clearly separated from the flight.
- **Front-facing (on-axis)**: the dart is mostly the flight; use `--tip-method centroid`. Least accurate.
- Lighting: a ring light or two lamps either side of the board; avoid a single overhead light that throws
  long dart shadows. Fix the camera rigidly – a bumped camera means recalibrating.
- Lock auto-exposure/white balance in your camera software if you can.

## Original bugs confirmed (approximate line refs into the original `dart_tracker.py`)
1. L109-133: every frame re-detects and prints the same dart; no state or deduping.
2. L144-147: baseline captured once; dart 2 and 3 frames also contain earlier darts, so they are re-scored.
3. L118-119: scores `minEnclosingCircle` centre (the barrel/flight), not the tip.
4. L76-81 vs L22-26: calibration puts the double wire at 350 px but scoring treats 340 px as the outer edge.
   Ring radii (12/30/190-210/320-340) don't match real proportions (treble ≈ 0.58–0.63 of double radius).
5. L28-33: `190 < r < 210` / `320 < r < 340` leave points exactly on 190, 210, 320, 340 falling into the wrong ring.
6. L14: `atan2(dy, dx) + 90` is correct only by chance with y-down coordinates; untested.
7. L81: `findHomography` with 4 points where 3 (bull, left, right) are collinear – a near-degenerate fit.
8. L39, L64: `cv2.resize` to 1280x720 always, never sets capture resolution; aspect distortion on 4:3 cameras.
9. L92, L101: `break` on one failed read ends the program; no reconnect.
10. No contour filtering beyond area; shadows and lighting drift trigger false scores. Magic numbers everywhere, `print` logging.

## Tuning guide (adjust in this order)
| Symptom | Parameter |
|---|---|
| Ghost darts, flicker, random scores | raise `--diff-threshold` (30→40), raise `--min-area` |
| Dart missed entirely / only partly seen | lower `--diff-threshold` (→20), lower `--min-aspect-ratio` |
| Scored before the dart stops wobbling, or a hand gets scored | raise `--settle-frames` (15→25) |
| Takes too long to score | lower `--settle-frames` |
| Arm in frame produces scores | lower `--hand-motion-area` |
| Dart never accepted with a close camera (big blob) | raise `--max-area`, raise `--hand-motion-area` |
| Shadow scored instead of dart | keep `--shadow-v-only true`, raise `--min-solidity`, add side lighting |
| Score is consistently a ring too far out/in | wrong `--camera-side` or try `--tip-method lowest` |
| Score off by a constant amount everywhere | recalibrate with 5 points; check the overlay |
| Slow light changes eventually trigger detections | raise `--background-alpha` (0.01→0.03) |
| Speckle noise in mask (`--debug`) | raise `--morph-kernel` to 5, raise `--blur-kernel` to 7 |

## Known limitations / next steps
- Single camera: tip depth is ambiguous, so darts close to a wire can land on the wrong side. 2–3 cameras with triangulation (as in commercial systems) fix this.
- Overlapping darts (robin-hooding, tight groupings) can merge into one blob.
- Tip estimation is a heuristic; a small ML keypoint model (e.g. YOLO-pose trained on dart tips) would be more accurate.
- Board rotation is assumed to be standard (20 at top); only the homography corrects small rotation.
- Auto-exposure changes can trigger false detections; lock camera exposure if possible.
- No persistence of game state between runs; no cricket or other game modes.


## Live diagnostic dashboard (v2.4)
The application now opens a live diagnostics window by default alongside the scorer windows. It shows both camera feeds (when enabled), calibration overlays, detected tip markers, a board-plane position preview, last calculated score and fusion method, per-camera FPS and detector processing time, detector state, tip-shape quality heuristic, camera read skew, and projected-tip separation when both cameras have detections.

Dashboard controls (available when the score-entry override is not active):
- `d`: show/hide the live diagnostics dashboard.
- `x`: increment the manually tracked false-positive count.
- `m`: increment the manually tracked missed-dart count.

The confidence/quality value is a heuristic based on contour shape and area; it is **not** a calibrated probability. The camera skew value is the difference between sequential frame-read completion times, not hardware exposure synchronisation. The missed/false-positive counters require the operator to annotate events with `m`/`x`; they are not inferred automatically.

## Visual offline diagnostics (added in diagnostics update)
Create an annotated video and a per-frame CSV from any recorded clip:
```bash
python -m dart_scorer.diagnostics session.mp4 --calibration calibration.json \
    --output diagnostics_dashboard.mp4 --csv diagnostics_frames.csv --baseline-frame 5 \
    --tip-method lowest --camera-side bottom --settle-frames 8 --diff-threshold 25
```
The dashboard shows the source frame with the calibration overlay and estimated tip, the difference mask, detector state, changed-pixel count, score, camera/board coordinates, contour area/aspect/solidity, and a heuristic shape-quality indicator. Each detection stays on screen for ~1 s and earlier tips remain marked; non-16:9 clips are letterboxed. `--baseline-frame` must lie inside the clip. That indicator is a debugging aid, not a calibrated probability of correctness. Add `--show` to display the dashboard while processing; press `q` to stop.
