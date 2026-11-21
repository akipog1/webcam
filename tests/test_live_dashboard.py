import numpy as np

from dart_scorer.config import DetectorConfig
from dart_scorer.detector import DartDetector
from dart_scorer.game import GameState
from dart_scorer.live_dashboard import LiveDashboard


def test_dashboard_renders_single_camera_layout():
    dashboard = LiveDashboard(False)
    detector = DartDetector(DetectorConfig())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    game = GameState.new(["Tester"], 501, True)
    rendered = dashboard.render(frame, None, detector, None, None, None, game, None)
    assert rendered.shape == (670, 960, 3)
    assert rendered.dtype == np.uint8


def test_manual_diagnostic_counters_and_result():
    dashboard = LiveDashboard()
    dashboard.false_positives += 1
    dashboard.missed_darts += 2
    dashboard.record_result((300.0, 300.0), "intersection")
    assert dashboard.false_positives == 1
    assert dashboard.missed_darts == 2
    assert dashboard._last_method == "intersection"


def test_dashboard_render_throttling():
    dashboard = LiveDashboard()
    assert dashboard.should_render(1.0)
    assert not dashboard.should_render(1.05)
    assert dashboard.should_render(1.10)
