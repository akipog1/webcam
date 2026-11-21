"""GUI-free game/detector glue: scoring, undo, overrides and key handling.

Everything the app does in response to a detection, key or click lives here so
it can be unit-tested without OpenCV windows or a camera.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .calibration import Calibrator
from .detector import DartDetector, Detection, State
from .game import GameState
from .scorer import Score, parse_score, score_point

log = logging.getLogger("dart_scorer")

EDIT_CHARS = "0123456789TDSB"


class Action(Enum):
    CONTINUE = "continue"
    QUIT = "quit"
    RECALIBRATE = "recalibrate"


class ThrowLogger:
    """Append each throw to CSV. A row with source ``undo`` voids the previous
    row for that player (it repeats the undone dart)."""

    FIELDS = [
        "timestamp",
        "player",
        "raw_x",
        "raw_y",
        "flat_x",
        "flat_y",
        "ring",
        "sector",
        "points",
        "label",
        "source",
    ]

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, player: str, raw, flat, s: Score, source: str = "auto") -> None:
        def fmt(p, i):
            return "" if p is None else f"{p[i]:.1f}"

        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(timespec="milliseconds"),
                    player,
                    fmt(raw, 0),
                    fmt(raw, 1),
                    fmt(flat, 0),
                    fmt(flat, 1),
                    s.ring,
                    s.sector or "",
                    s.points,
                    s.label,
                    source,
                ]
            )


class Controller:
    def __init__(
        self, game: GameState, det: DartDetector, cal: Calibrator, logger: ThrowLogger
    ) -> None:
        self.game, self.det, self.cal, self.logger = game, det, cal, logger
        #: second detector in two-camera mode; kept in step with ``det`` on b/n/undo
        self.det2: Optional[DartDetector] = None
        self.message = "Press 'b' with an EMPTY board"
        self.edit_buffer: Optional[str] = None
        #: one entry per accepted dart this turn; position is None for typed scores
        self.marks: list[tuple[Optional[tuple[float, float]], Score]] = []

    # ---------- scoring ----------
    def _accept(self, s: Score) -> bool:
        """Add a dart to the game. Returns False (and sets the message) if the game refused it."""
        before = len(self.game.turn_darts)
        r = self.game.add_dart(s)
        if len(self.game.turn_darts) == before:
            self.message = f"Ignored {s.label}: {r.message}"
            return False
        self.message = f"{s.label} ({s.points})  {r.message}"
        return True

    def on_detection(self, d: Detection) -> Optional[Score]:
        return self.score(d.tip, self.cal.to_board(d.tip))

    def on_fused(self, tip) -> Optional[Score]:
        """Score a two-camera fused tip (``fusion.FusedTip``)."""
        return self.score(tip.raw, tip.flat, source=f"auto-{tip.method}")

    def score(self, raw, flat, source: str = "auto") -> Optional[Score]:
        s = score_point(*flat)
        if not self._accept(s):
            return None
        self.marks.append((flat, s))
        self.logger.log(self.game.player.name, raw, flat, s, source)
        log.info("%s: %s", self.game.player.name, s.label)
        return s

    def board_click(self, x: float, y: float) -> None:
        """Re-score the last dart at a clicked point in the flat board view."""
        if not self.game.turn_darts:
            return
        s = score_point(x, y)
        self.game.override_last(s)
        self.marks[-1:] = [((x, y), s)]
        self.logger.log(self.game.player.name, None, (x, y), s, "click_override")
        self.message = f"Corrected to {s.label}"

    def undo(self, frame: np.ndarray, frame2: Optional[np.ndarray] = None) -> bool:
        if not self.game.turn_darts:
            return False
        gone = self.game.turn_darts[-1]
        self.game.undo()
        if self.marks:
            self.marks.pop()
        self.det.undo_dart(frame)
        if self.det2 is not None and frame2 is not None:
            self.det2.undo_dart(frame2)
        self.logger.log(self.game.player.name, None, None, gone, "undo")
        self.message = "Undid last dart"
        return True

    # ---------- keys ----------
    def handle_key(self, k: int, frame: np.ndarray, frame2: Optional[np.ndarray] = None) -> Action:
        if self.edit_buffer is not None:
            self._edit_key(k)
            return Action.CONTINUE
        if k == ord("q"):
            return Action.QUIT
        if k == ord("b"):
            self.det.capture_baseline(frame)
            if self.det2 is not None and frame2 is not None:
                self.det2.capture_baseline(frame2)
            self.message = "Ready - throw!"
        elif k == ord("n"):
            self.game.next_turn()
            self.det.next_turn(frame)
            if self.det2 is not None and frame2 is not None:
                self.det2.next_turn(frame2)
            self.marks.clear()
            self.message = f"{self.game.player.name} to throw"
        elif k == ord("u"):
            self.undo(frame, frame2)
        elif k == ord("e"):
            self.edit_buffer = ""
        elif k == ord("c"):
            return Action.RECALIBRATE
        return Action.CONTINUE

    def after_recalibrate(self) -> None:
        self.det.state = State.WAIT_BASELINE
        if self.det2 is not None:
            self.det2.state = State.WAIT_BASELINE
        self.message = "Recalibrated - press 'b'"

    def _edit_key(self, k: int) -> None:
        if k in (13, 10):
            try:
                s = parse_score(self.edit_buffer)
            except ValueError:
                self.message = "Invalid entry"
            else:
                if self.game.turn_darts:
                    self.game.override_last(s)
                    self.marks[-1:] = [(self.marks[-1][0] if self.marks else None, s)]
                    ok = True
                else:
                    ok = self._accept(s)
                    if ok:
                        self.marks.append((None, s))
                if ok:
                    self.logger.log(self.game.player.name, None, None, s, "key_override")
                    self.message = f"Set to {s.label}"
            self.edit_buffer = None
        elif k == 27:
            self.edit_buffer = None
        elif k == 8:
            self.edit_buffer = self.edit_buffer[:-1]
        elif 0 <= k < 128 and chr(k).upper() in EDIT_CHARS:
            self.edit_buffer += chr(k).upper()
