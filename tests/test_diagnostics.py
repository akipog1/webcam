import numpy as np

from dart_scorer.config import DetectorConfig
from dart_scorer.diagnostics import _quality, _shape_metrics, render_dashboard_frame
from dart_scorer.calibration import Calibrator
from dart_scorer.detector import DartDetector


def test_shape_metrics_and_quality_are_bounded():
    contour = np.array([[[0, 0]], [[30, 0]], [[30, 5]], [[0, 5]]], dtype=np.int32)
    area, aspect, solidity = _shape_metrics(contour)
    assert area > 0 and aspect > 1 and 0 < solidity <= 1
    q = _quality(area, aspect, solidity, DetectorConfig())
    assert 0 <= q <= 100


def test_dashboard_frame_dimensions():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cal = Calibrator()
    # A simple affine mapping sufficient for rendering the diagnostic frame.
    cal.set_matrix(np.array([[1000/1280, 0, 0], [0, 1000/720, 0], [0, 0, 1]], dtype=float))
    cal.frame_size = (1280, 720)
    det = DartDetector(DetectorConfig())
    image = render_dashboard_frame(frame, np.zeros((720,1280),np.uint8), cal, det, 1, None, 30, det.cfg)
    assert image.shape == (720, 1280, 3)


def test_dashboard_keeps_aspect_ratio_for_4x3_clip():
    from dart_scorer.diagnostics import _fit
    out = _fit(np.full((480, 640, 3), 255, np.uint8), 800, 450)
    assert out.shape == (450, 800, 3)
    cols = np.where(out.any(axis=(0, 2)))[0]
    assert abs((cols[-1] - cols[0] + 1) - 600) <= 1      # 4:3 at 450 px high -> 600 px wide


def test_bad_baseline_frame_rejected(tmp_path):
    import cv2, pytest
    from dart_scorer.diagnostics import run_diagnostics
    clip = str(tmp_path / "c.mp4")
    w = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for _ in range(5):
        w.write(np.zeros((720, 1280, 3), np.uint8))
    w.release()
    cal = Calibrator(tmp_path / "cal.json")
    cal.set_matrix(np.eye(3)); cal.frame_size = (1280, 720); cal.save()
    for bad in (0, 99):
        with pytest.raises(ValueError):
            run_diagnostics(clip, str(tmp_path / "cal.json"), str(tmp_path / "o.mp4"),
                            str(tmp_path / "o.csv"), bad, DetectorConfig())
