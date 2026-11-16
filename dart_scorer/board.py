"""Single source of truth for dartboard geometry.

All radii derive from regulation steel-tip board dimensions (millimetres) and
are scaled into the flat, top-down canvas used after homography. Both the
calibration targets and the scorer import from here, so they can never drift
apart (the original code calibrated to 350 px but scored the double at 340 px).
"""
from __future__ import annotations

import math

CANVAS_SIZE: int = 1000
CENTER: tuple[float, float] = (CANVAS_SIZE / 2.0, CANVAS_SIZE / 2.0)

INNER_BULL_MM = 6.35
OUTER_BULL_MM = 15.9
TREBLE_INNER_MM = 99.0
TREBLE_OUTER_MM = 107.0
DOUBLE_INNER_MM = 162.0
DOUBLE_OUTER_MM = 170.0

#: Double outer wire sits at this canvas radius; leaves margin for misses.
DOUBLE_OUTER_PX: float = 400.0
PX_PER_MM: float = DOUBLE_OUTER_PX / DOUBLE_OUTER_MM

INNER_BULL_PX = INNER_BULL_MM * PX_PER_MM
OUTER_BULL_PX = OUTER_BULL_MM * PX_PER_MM
TREBLE_INNER_PX = TREBLE_INNER_MM * PX_PER_MM
TREBLE_OUTER_PX = TREBLE_OUTER_MM * PX_PER_MM
DOUBLE_INNER_PX = DOUBLE_INNER_MM * PX_PER_MM

RING_RADII_PX: tuple[float, ...] = (
    INNER_BULL_PX, OUTER_BULL_PX, TREBLE_INNER_PX,
    TREBLE_OUTER_PX, DOUBLE_INNER_PX, DOUBLE_OUTER_PX,
)

#: Clockwise from the top.
SECTORS: tuple[int, ...] = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                            3, 19, 7, 16, 8, 11, 14, 9, 12, 5)


def polar_to_canvas(radius_px: float, angle_deg_clockwise_from_top: float) -> tuple[float, float]:
    """Convert board polar coordinates to canvas (image, y-down) coordinates."""
    a = math.radians(angle_deg_clockwise_from_top)
    return CENTER[0] + radius_px * math.sin(a), CENTER[1] - radius_px * math.cos(a)


def sector_wire_angle(sector_index: int) -> float:
    """Clockwise angle (deg from top) of the wire on the *left* edge of a sector."""
    return (sector_index * 18.0 - 9.0) % 360.0


def calibration_targets() -> dict[str, tuple[float, float]]:
    """Named canvas targets the user can click during calibration.

    Default set: bull centre plus the outer double wire at the middle of the
    20, 6, 3 and 11 sectors (top, right, bottom, left).
    """
    return {
        "bull": CENTER,
        "D20 outer wire (top)": polar_to_canvas(DOUBLE_OUTER_PX, 0),
        "D6 outer wire (right)": polar_to_canvas(DOUBLE_OUTER_PX, 90),
        "D3 outer wire (bottom)": polar_to_canvas(DOUBLE_OUTER_PX, 180),
        "D11 outer wire (left)": polar_to_canvas(DOUBLE_OUTER_PX, 270),
    }
