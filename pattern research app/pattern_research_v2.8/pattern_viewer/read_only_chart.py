from __future__ import annotations

import pandas as pd
from pyqtgraph.Qt import QtCore, QtWidgets

from pattern_analyzer.chart_widget import ChartWidget


class ReadOnlyChartWidget(ChartWidget):
    """Pattern Analyzer chart reused as a chart-read-only Pattern Viewer.

    View-only operations such as pan/zoom, timeframe, timezone, Auto/AutoAll,
    screenshot, drawing visibility and N-day statistics remain available.
    Drawing creation/editing and Drawing clipboard actions are disabled.

    Viewer-specific Measure is permanently enabled: right-button drag measures a
    transient price/percentage move, and releasing the button clears the overlay
    while keeping the next right-drag immediately available.
    """

    def __init__(self, parent=None):
        # B-mode: Drawings are born non-interactive inside ChartWidget.  Viewer no
        # longer creates editable ROI objects and then tries to lock them afterward.
        super().__init__(parent, drawing_interaction_enabled=False)

        # Hide drawing-creation tools. The Drawing visibility toggle remains useful.
        for widget in (self.btn_h, self.btn_l, self.btn_t, self.btn_rect, self.btn_fibo):
            widget.hide()

        # Viewer is always in Measure mode; there is no middle-button toggle.
        self.measure_status.setText("MEASURE")
        self.measure_status.setToolTip("Viewer：右鍵按住拖曳即可量測；放開後量測消失")
        self._set_measure_mode(True)
        self.measure_status.show()

        # Viewer must never expose Drawing copy/paste mutations.
        try:
            self.shortcut_copy_drawing.setEnabled(False)
            self.shortcut_paste_drawing.setEnabled(False)
        except Exception:
            pass

        self.show_drawings_checkbox.setToolTip("圖表唯讀：顯示 / 隱藏 Case 中已保存的 Drawing")

    def set_tool(self, name: str | None):
        # Drawing creation is intentionally unavailable in Pattern Viewer.
        self.tool_mode = None
        self.pending_point = None

    def _scene_clicked(self, evt):
        # Do not select/edit Drawings or open Drawing context menus.
        return

    def eventFilter(self, watched, event):
        """Permanent Viewer Measure: every right-button drag is a measurement."""
        try:
            viewport = self.graphics.viewport()
        except Exception:
            viewport = None
        if watched is not viewport:
            return QtWidgets.QWidget.eventFilter(self, watched, event)

        et = event.type()
        mouse_press = QtCore.QEvent.MouseButtonPress
        mouse_release = QtCore.QEvent.MouseButtonRelease
        mouse_move = QtCore.QEvent.MouseMove

        # Never toggle measure mode in Viewer. It is structurally always enabled.
        if et == mouse_press and event.button() == QtCore.Qt.RightButton:
            scene_pos = self._viewport_event_to_scene(event)
            self._measure_begin(scene_pos)
            event.accept()
            return True

        if et == mouse_move and self._measure_dragging:
            try:
                right_down = bool(event.buttons() & QtCore.Qt.RightButton)
            except Exception:
                right_down = True
            if right_down:
                scene_pos = self._viewport_event_to_scene(event)
                if scene_pos is not None:
                    point = self.plot.vb.mapSceneToView(scene_pos)
                    self._update_measure_overlay(float(point.x()), float(point.y()))
                event.accept()
                return True

        if et == mouse_release and event.button() == QtCore.Qt.RightButton:
            self._clear_measure_overlay()
            # _clear_measure_overlay releases the mouse but must not disable the mode.
            self.measure_mode = True
            try:
                self.graphics.viewport().setCursor(QtCore.Qt.CrossCursor)
            except Exception:
                pass
            event.accept()
            return True

        # Bypass ChartWidget.eventFilter so middle-click can never disable Measure.
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
