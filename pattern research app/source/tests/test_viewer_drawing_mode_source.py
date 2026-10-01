from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHART = (ROOT / "pattern_analyzer" / "chart_widget.py").read_text(encoding="utf-8")
VIEWER = (ROOT / "pattern_viewer" / "read_only_chart.py").read_text(encoding="utf-8")
VIEWER_WINDOW = (ROOT / "pattern_viewer" / "main_window.py").read_text(encoding="utf-8")


def test_viewer_constructs_chart_in_noninteractive_mode():
    assert "super().__init__(parent, drawing_interaction_enabled=False)" in VIEWER
    assert "_lock_all_drawing_items" not in VIEWER


def test_base_chart_has_explicit_drawing_interaction_mode():
    assert "drawing_interaction_enabled: bool = True" in CHART
    assert "self.drawing_interaction_enabled = bool(drawing_interaction_enabled)" in CHART


def test_viewer_mode_does_not_register_drawing_hit_targets():
    block = CHART[CHART.index("def _register_drawing_hit_item"):]
    assert "if not self.drawing_interaction_enabled:" in block[:300]


def test_text_rectangle_and_fibo_build_interaction_conditionally():
    assert "movable=self.drawing_interaction_enabled" in CHART
    assert "if self.drawing_interaction_enabled:\n            self._configure_right_bottom_resize_handles(roi)" in CHART
    assert "if self.drawing_interaction_enabled:\n            self._configure_four_side_resize_handles(roi)" in CHART
    assert "if self.drawing_interaction_enabled:\n            transparent_pen" in CHART


def test_viewer_order_toggle_lives_in_overlay_and_not_chart_toolbar():
    assert 'self.show_orders_checkbox = QtWidgets.QCheckBox("全部 Order")' in VIEWER_WINDOW
    assert "self.show_orders_checkbox.toggled.connect(self._update_order_visibility)" in VIEWER_WINDOW
    assert "self.show_orders_checkbox.hide()" in VIEWER
