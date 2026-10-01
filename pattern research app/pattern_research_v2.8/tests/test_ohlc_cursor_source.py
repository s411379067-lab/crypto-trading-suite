from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHART = (ROOT / "pattern_analyzer" / "chart_widget.py").read_text(encoding="utf-8")


def test_mouse_move_updates_ohlc_from_nearest_rendered_candle():
    block = CHART[CHART.index("def _mouse_moved"):CHART.index("def _update_crosshair_coordinate_labels")]
    assert 'hovered_row = self._last_bars.iloc[idx]' in block
    assert 'self._set_ohlc_row(hovered_row)' in block


def test_cursor_leave_restores_latest_rendered_candle():
    block = CHART[CHART.index("def _mouse_moved"):CHART.index("def _update_crosshair_coordinate_labels")]
    assert 'self._restore_latest_ohlc()' in block


def test_render_defaults_ohlc_to_latest_visible_candle():
    render = CHART[CHART.index("def render("):CHART.index("def set_all_drawings_visible")]
    assert 'self._set_ohlc_row(bars.iloc[-1])' in render
