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


def test_viewer_selected_case_shows_realized_pnl():
    assert 'self.selected_pnl_label = QtWidgets.QLabel("Realized PnL: --")' in VIEWER_WINDOW
    assert "self._set_selected_realized_pnl(realized_pnl)" in VIEWER_WINDOW
    assert 'color, value = "#7bd88f"' in VIEWER_WINDOW
    assert 'color, value = "#ff6b6b"' in VIEWER_WINDOW


def test_viewer_exposes_pattern_presence_and_exclusion_filters():
    assert '("有 Pattern", "HAS_PATTERN")' in VIEWER_WINDOW
    assert 'row["operator"].addItem("不包含任一", "EXCLUDE_ANY")' in VIEWER_WINDOW
    assert 'row["operator"].addItem("不包含全部", "EXCLUDE_ALL")' in VIEWER_WINDOW
    assert 'self.combine_combo.addItem("AND — 全部符合", "AND")' in VIEWER_WINDOW
    assert 'self.combine_combo.addItem("OR — 任一符合", "OR")' in VIEWER_WINDOW
    assert 'self.btn_add_filter = QtWidgets.QPushButton("+ Filter")' in VIEWER_WINDOW


def test_viewer_filter_selection_and_side_panels_are_resizable():
    assert "self.selected_rows_layout.addWidget(row)" in VIEWER_WINDOW
    assert "self.filter_rows_scroll.setMinimumHeight(280)" in VIEWER_WINDOW
    assert "self.sidebar_splitter.setChildrenCollapsible(False)" in VIEWER_WINDOW
    assert "self.case_content_splitter.setChildrenCollapsible(False)" in VIEWER_WINDOW
    assert "self.inspector_splitter.setChildrenCollapsible(False)" in VIEWER_WINDOW
    assert "splitter.setHandleWidth(7)" in VIEWER_WINDOW
