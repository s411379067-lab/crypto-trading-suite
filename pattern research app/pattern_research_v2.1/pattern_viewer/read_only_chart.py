from __future__ import annotations

import pandas as pd
from pyqtgraph.Qt import QtCore, QtWidgets

from pattern_analyzer.chart_widget import ChartWidget


class ReadOnlyChartWidget(ChartWidget):
    """Pattern Analyzer chart reused as a strictly non-editing viewer.

    View-only operations such as pan/zoom, timeframe, timezone, Auto/AutoAll,
    screenshot, drawing visibility and N-day statistics remain available.
    Drawing creation/editing, Measure Mode, context menus and clipboard actions
    are disabled. No Case save path is exposed by Pattern Viewer.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Hide drawing-creation tools. The Drawing visibility toggle remains useful.
        for widget in (self.btn_h, self.btn_l, self.btn_t, self.btn_rect, self.btn_fibo):
            widget.hide()
        self.measure_status.hide()

        # Viewer must never expose Drawing copy/paste mutations.
        try:
            self.shortcut_copy_drawing.setEnabled(False)
            self.shortcut_paste_drawing.setEnabled(False)
        except Exception:
            pass

        self.show_drawings_checkbox.setToolTip("唯讀檢視：顯示 / 隱藏 Case 中已保存的 Drawing")

    def set_tool(self, name: str | None):
        # Drawing creation is intentionally unavailable in Pattern Viewer.
        self.tool_mode = None
        self.pending_point = None

    def _scene_clicked(self, evt):
        # Do not select/edit Drawings or open Drawing context menus.
        return

    def eventFilter(self, watched, event):
        # Skip ChartWidget's middle-click Measure Mode / right-drag measurement.
        # PyQtGraph keeps its ordinary pan/zoom behavior.
        return QtWidgets.QWidget.eventFilter(self, watched, event)

    def _tf_changed(self, tf: str):
        # View-only: do not mutate case.display or emit dirty.
        self.render(reset_x=False)
        self.view_timeframe_changed.emit(tf)

    def _x_tick_changed(self, interval: str):
        # View-only: update axis only.
        self.axis.set_tick_interval(interval)
        self.axis.picture = None
        self.axis.update()

    def _tz_changed(self, tz: str):
        # View-only: change rendering timezone without touching Case JSON.
        if not tz:
            return
        try:
            pd.Timestamp.now(tz=tz)
        except Exception:
            return
        self.axis.set_timezone(tz)
        self.timezone_changed.emit(tz)

    def _mouse_moved(self, evt):
        # Keep crosshair/coordinate display but never expose Order S/L buttons.
        super()._mouse_moved(evt)
        try:
            self.btn_short_axis.hide()
            self.btn_long_axis.hide()
        except Exception:
            pass

    def _refresh_selection_visuals(self):
        # Viewer has no selected Drawing state and therefore no resize handles.
        self.selected_drawing_id = None
        super()._refresh_selection_visuals()
        self._lock_all_drawing_items()

    def _render_drawing_items(self):
        super()._render_drawing_items()
        self._lock_all_drawing_items()

    def _lock_roi(self, roi):
        if roi is None:
            return
        try:
            roi.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        except Exception:
            pass
        try:
            self._set_roi_handles_visible(roi, False)
        except Exception:
            pass
        for marker in list(getattr(roi, "_standard_handle_markers", []) or []):
            try:
                marker.setVisible(False)
                marker.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            except Exception:
                pass
        try:
            handles = list(roi.getHandles())
        except Exception:
            handles = []
        for handle in handles:
            try:
                handle.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            except Exception:
                pass

    def _lock_all_drawing_items(self):
        """Disable every mutable Drawing hit target after each render/rebuild."""
        self.selected_drawing_id = None
        for item in list(self.drawing_items.values()):
            if isinstance(item, dict):
                roi = item.get("roi")
                self._lock_roi(roi)
                for key in ("text", "border", "selection", "background", "fill", "outline"):
                    obj = item.get(key)
                    if obj is not None:
                        try:
                            obj.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                        except Exception:
                            pass
                for key in ("anchors", "anchor_markers", "hit_lines", "lines"):
                    for obj in item.get(key, []) or []:
                        try:
                            obj.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                        except Exception:
                            pass
                        if key in {"anchors", "anchor_markers"}:
                            try:
                                obj.setVisible(False)
                            except Exception:
                                pass
                continue

            self._lock_roi(item)
            try:
                item.setMovable(False)
            except Exception:
                pass
            try:
                item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            except Exception:
                pass
