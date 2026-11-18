"""Pure scoring functions. No OpenCV, fully unit-testable.

Boundary convention (contiguous, no gaps): a point exactly on a wire belongs
to the *inner* region, i.e. every ring is ``(inner, outer]``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from . import board


@dataclass(frozen=True)
class Score:
    """Result of scoring one dart."""
    label: str
    points: int
    ring: str  # inner_bull | outer_bull | inner_single | treble | outer_single | double | miss
    sector: Optional[int]
    multiplier: int  # 0 miss/outer bull(single 25), 1 single, 2 double (incl. bull), 3 treble

    @property
    def is_double(self) -> bool:
        return self.multiplier == 2


def angle_clockwise_from_top(x: float, y: float) -> float:
    """Angle in [0, 360) measured clockwise from 12 o'clock (canvas is y-down)."""
    dx, dy = x - board.CENTER[0], y - board.CENTER[1]
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def sector_at_angle(angle: float) -> int:
    """Sector for a clockwise-from-top angle; sector 20 spans [-9, 9)."""
    return board.SECTORS[int(((angle + 9.0) % 360.0) // 18.0)]


def score_polar(radius: float, angle: float) -> Score:
    """Score a point given canvas radius (px) and clockwise angle (deg)."""
    if radius <= board.INNER_BULL_PX:
        return Score("Bullseye", 50, "inner_bull", None, 2)
    if radius <= board.OUTER_BULL_PX:
        return Score("Outer bull", 25, "outer_bull", None, 1)
    if radius > board.DOUBLE_OUTER_PX:
        return Score("Miss", 0, "miss", None, 0)
    s = sector_at_angle(angle)
    if radius <= board.TREBLE_INNER_PX:
        return Score(f"S{s}", s, "inner_single", s, 1)
    if radius <= board.TREBLE_OUTER_PX:
        return Score(f"T{s}", s * 3, "treble", s, 3)
    if radius <= board.DOUBLE_INNER_PX:
        return Score(f"S{s}", s, "outer_single", s, 1)
    return Score(f"D{s}", s * 2, "double", s, 2)


def score_point(x: float, y: float) -> Score:
    """Score flat-canvas coordinates."""
    r = math.hypot(x - board.CENTER[0], y - board.CENTER[1])
    return score_polar(r, angle_clockwise_from_top(x, y))


def parse_score(text: str) -> Score:
    """Parse a manual override like ``T20``, ``D16``, ``S5``, ``25``, ``50``/``B`` (bull), ``0``."""
    t = text.strip().upper()
    if t in ("50", "B", "BULL", "DB"):
        return Score("Bullseye", 50, "inner_bull", None, 2)
    if t in ("25", "SB"):
        return Score("Outer bull", 25, "outer_bull", None, 1)
    if t in ("0", "MISS", "M"):
        return Score("Miss", 0, "miss", None, 0)
    mult = {"S": 1, "D": 2, "T": 3}.get(t[:1])
    try:
        num = int(t[1:] if mult else t)
    except ValueError:
        raise ValueError(f"cannot parse score {text!r}") from None
    mult = mult or 1
    if not 1 <= num <= 20:
        raise ValueError(f"invalid sector in {text!r}")
    ring = {1: "outer_single", 2: "double", 3: "treble"}[mult]
    return Score(f"{'SDT'[mult - 1]}{num}", num * mult, ring, num, mult)
