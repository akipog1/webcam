"""Physically consistent two-camera synthetic scene.

World: millimetres, board centre at the origin, X right, Y down (as on the
canvas), Z = 0 on the board face and negative toward the thrower. Cameras are
pinhole models; a dart is a 3-D segment from the tip (on the board) outward.
Because the projection is real, a dart's flight lands in a different place for
each camera, which is exactly what makes single-camera scoring ambiguous.
"""
import cv2
import numpy as np

from dart_scorer import board
from dart_scorer.calibration import Calibrator
from synth import H, W, render_board

F = 900.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], np.float64)
CAM1 = (0.0, 450.0, -520.0)       # below and in front of the board
CAM2 = (-560.0, 60.0, -420.0)     # to the left, slightly low


def _pose(pos):
    c = np.array(pos, np.float64)
    f = -c / np.linalg.norm(c)                        # look at the board centre
    down = np.array([0.0, 1.0, 0.0]) - f * f[1]
    down /= np.linalg.norm(down)
    x = np.cross(down, f)
    R = np.stack([x, down, f])
    t = -R @ c
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1)


class Cam:
    def __init__(self, pos):
        self.rvec, self.tvec = _pose(pos)
        # canvas px -> image px homography for the board plane (Z = 0)
        cv = np.array([[200, 200], [800, 200], [800, 800], [200, 800], [500, 500]], np.float32)
        img = self.project(np.array([[self.mm(p)[0], self.mm(p)[1], 0.0] for p in cv]))
        self.h_canvas2cam, _ = cv2.findHomography(cv, img.astype(np.float32))

    @staticmethod
    def mm(canvas_pt):
        return ((canvas_pt[0] - board.CENTER[0]) / board.PX_PER_MM,
                (canvas_pt[1] - board.CENTER[1]) / board.PX_PER_MM)

    def project(self, pts3):
        out, _ = cv2.projectPoints(np.asarray(pts3, np.float64).reshape(-1, 1, 3),
                                   self.rvec, self.tvec, K, None)
        return out.reshape(-1, 2)

    def background(self):
        return render_board(self.h_canvas2cam)

    def calibrated(self, noise=0.8, seed=0):
        rng = np.random.default_rng(seed)
        cal = Calibrator()
        names = list(cal.targets)
        clicks = []
        for n in names:
            x, y = self.mm(cal.targets[n])
            clicks.append(tuple(self.project([[x, y, 0.0]])[0] + rng.uniform(-noise, noise, 2)))
        cal.fit(clicks, [cal.targets[n] for n in names])
        cal.frame_size = (W, H)
        return cal

    def add_dart(self, frame, canvas_tip, length_mm=140.0, lean=(0.10, -0.22)):
        """Draw a dart whose tip is at ``canvas_tip`` on the board and whose flight points at the thrower."""
        x, y = self.mm(canvas_tip)
        d = np.array([lean[0], lean[1], -1.0])
        d /= np.linalg.norm(d)
        tip = np.array([x, y, 0.0])
        a, b = self.project([tip, tip + d * length_mm])
        cv2.line(frame, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b),
                 (255, 255, 255), 5, cv2.LINE_AA)
