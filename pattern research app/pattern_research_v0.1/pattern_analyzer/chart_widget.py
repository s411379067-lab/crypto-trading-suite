from __future__ import annotations

from typing import Callable
import uuid
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.aggregation import aggregate_visible_bars, timeframe_seconds


class TimeAxis(pg.AxisItem):
    def __init__(self, *args, timezone_name="Asia/Taipei", **kwargs):
        super().__init__(*args, **kwargs)
        self.timezone_name = timezone_name

    def set_timezone(self, timezone_name: str):
        self.timezone_name = timezone_name
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        labels = []
        for ts in values:
            try:
                dt = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(self.timezone_name)
                if spacing >= 86400:
                    labels.append(dt.strftime("%Y.%m.%d"))
                else:
                    labels.append(f"{dt.strftime('%H:%M')}\n({dt.strftime('%m.%d')})")
            except Exception:
                labels.append("")
        return labels



class MovableTextItem(pg.TextItem):
    """Text item that notifies the domain layer after a drag finishes."""
    movementFinished = QtCore.Signal()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.movementFinished.emit()


class LineSettingsDialog(QtWidgets.QDialog):
    STYLE_OPTIONS = [
        ("實線", "solid"),
        ("虛線", "dashed"),
        ("點線", "dotted"),
    ]

    def __init__(self, color: str, style_name: str, width_value: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("線條設定")
        self.resize(350, 205)
        self._color = QtGui.QColor(color if color else "#00ffff")

        layout = QtWidgets.QVBoxLayout(self)

        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(QtWidgets.QLabel("顏色:"))
        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setFixedWidth(125)
        self.color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self.color_btn)
        color_row.addStretch(1)
        layout.addLayout(color_row)

        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(QtWidgets.QLabel("線型:"))
        self.style_combo = QtWidgets.QComboBox()
        for label, key in self.STYLE_OPTIONS:
            self.style_combo.addItem(label, key)
        idx = self.style_combo.findData(style_name or "solid")
        self.style_combo.setCurrentIndex(idx if idx >= 0 else 0)
        style_row.addWidget(self.style_combo)
        style_row.addStretch(1)
        layout.addLayout(style_row)

        width_row = QtWidgets.QHBoxLayout()
        width_row.addWidget(QtWidgets.QLabel("粗度:"))
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 12)
        self.width_spin.setValue(max(1, int(width_value or 2)))
        width_row.addWidget(self.width_spin)
        width_row.addStretch(1)
        layout.addLayout(width_row)

        buttons = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("套用")
        cancel_btn = QtWidgets.QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        self._refresh_color_button()

    def _pick_color(self):
        color = QtWidgets.QColorDialog.getColor(self._color, self, "選擇線條顏色")
        if color.isValid():
            self._color = color
            self._refresh_color_button()

    def _refresh_color_button(self):
        self.color_btn.setText(self._color.name().upper())
        self.color_btn.setStyleSheet(
            f"background-color:{self._color.name()}; border:1px solid #666; color:#ffffff;"
        )

    def values(self):
        return self._color.name(), str(self.style_combo.currentData()), int(self.width_spin.value())


class ChartWidget(QtWidgets.QWidget):
    dirty = QtCore.Signal()
    view_timeframe_changed = QtCore.Signal(str)
    timezone_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_df = pd.DataFrame()
        self.replay = None
        self.case = None
        self.current_case_path = None
        self.drawing_items: dict[str, object] = {}
        self.tool_mode: str | None = None
        self.pending_point = None
        self._last_bars = pd.DataFrame()
        self.selected_drawing_id: str | None = None
        self.auto_all_mode = False
        self._syncing_auto_all = False

        pg.setConfigOption("background", "#181c27")
        pg.setConfigOption("foreground", "white")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)

        self.toolbar = QtWidgets.QWidget()
        tb = QtWidgets.QHBoxLayout(self.toolbar)
        tb.setContentsMargins(2, 2, 2, 2)
        tb.setSpacing(5)

        self.btn_h = QtWidgets.QPushButton("H")
        self.btn_l = QtWidgets.QPushButton("L")
        self.btn_t = QtWidgets.QPushButton("T")
        self.btn_shot = QtWidgets.QPushButton("Shot")
        self.btn_auto = QtWidgets.QPushButton("Auto")
        self.btn_auto_all = QtWidgets.QPushButton("AutoAll")
        for b in (self.btn_h, self.btn_l, self.btn_t):
            b.setFixedSize(40, 30)
        self.btn_shot.setFixedSize(60, 30)
        self.btn_auto.setFixedSize(60, 30)
        self.btn_auto_all.setFixedSize(70, 30)

        self.timeframe_combo = QtWidgets.QComboBox()
        self.timeframe_combo.addItems(["M1", "M5", "M15", "H1"])
        self.timeframe_combo.setFixedWidth(90)

        self.timezone_combo = QtWidgets.QComboBox()
        self.timezone_combo.setEditable(True)
        self.timezone_combo.addItems(["Asia/Taipei", "America/New_York", "UTC", "Europe/London"])
        self.timezone_combo.setFixedWidth(170)

        for w in (self.btn_h, self.btn_l, self.btn_t, self.btn_shot, self.btn_auto, self.btn_auto_all,
                  self.timeframe_combo, self.timezone_combo):
            tb.addWidget(w)
        tb.addStretch(1)
        outer.addWidget(self.toolbar)

        self.info_panel = QtWidgets.QWidget()
        self.info_panel.setStyleSheet("background-color:#121722; border:1px solid #2a3142; border-radius:4px;")
        il = QtWidgets.QVBoxLayout(self.info_panel)
        il.setContentsMargins(10, 5, 10, 5)
        self.symbol_label = QtWidgets.QLabel("No Case")
        self.symbol_label.setStyleSheet("font-size:12pt; font-weight:700; color:#f5f7fa;")
        self.ohlc_label = QtWidgets.QLabel("開=--  高=--  低=--  收=--")
        self.ohlc_label.setStyleSheet("font-size:11pt; font-weight:700; color:#cfd7e6;")
        il.addWidget(self.symbol_label)
        il.addWidget(self.ohlc_label)
        outer.addWidget(self.info_panel)

        self.graphics = pg.GraphicsLayoutWidget()
        self.axis = TimeAxis(orientation="bottom", timezone_name="Asia/Taipei")
        self.plot = self.graphics.addPlot(axisItems={"bottom": self.axis}, row=0, col=0)
        self.plot.showAxis("right", True)
        self.plot.showAxis("left", False)
        self.plot.getAxis("right").setWidth(72)
        self.plot.showGrid(x=False, y=True, alpha=0.25)
        self.plot.hideButtons()
        outer.addWidget(self.graphics, 1)

        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen((150, 150, 150), width=1))
        self.hline = pg.InfiniteLine(angle=0, pen=pg.mkPen((150, 150, 150), width=1))
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot.addItem(self.hline, ignoreBounds=True)
        self.vline.hide()
        self.hline.hide()

        self.btn_h.clicked.connect(lambda: self.set_tool("horizontal_line"))
        self.btn_l.clicked.connect(lambda: self.set_tool("trend_line"))
        self.btn_t.clicked.connect(lambda: self.set_tool("text"))
        self.btn_auto.clicked.connect(self.auto_scale)
        self.btn_auto_all.clicked.connect(self.auto_all)
        self.btn_shot.clicked.connect(self.export_screenshot)
        self.timeframe_combo.currentTextChanged.connect(self._tf_changed)
        self.timezone_combo.currentTextChanged.connect(self._tz_changed)
        self.plot.scene().sigMouseClicked.connect(self._scene_clicked)
        self.plot.sigRangeChanged.connect(self._on_view_range_changed)
        self._mouse_proxy = pg.SignalProxy(self.graphics.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def set_context(self, case, raw_df: pd.DataFrame, replay, case_path: str):
        self.case = case
        self.raw_df = raw_df
        self.replay = replay
        self.current_case_path = case_path
        self.symbol_label.setText(f"{case.case.get('symbol', '')}   {case.case.get('research_date', '')}")

        self.timeframe_combo.blockSignals(True)
        self.timeframe_combo.setCurrentText(case.display.get("view_timeframe", "M5"))
        self.timeframe_combo.blockSignals(False)
        self.timezone_combo.blockSignals(True)
        self.timezone_combo.setCurrentText(case.display.get("timezone", "Asia/Taipei"))
        self.timezone_combo.blockSignals(False)
        self.axis.set_timezone(self.timezone_combo.currentText())

        self.rebuild_drawings()
        self.render(reset_x=True)

    def set_tool(self, name: str | None):
        self.tool_mode = name
        self.pending_point = None
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Drawing: {name}")

    def _tf_changed(self, tf: str):
        if self.case is not None:
            self.case.display["view_timeframe"] = tf
            self.case.touch()
            self.dirty.emit()
        self.render(reset_x=False)
        self.view_timeframe_changed.emit(tf)

    def _tz_changed(self, tz: str):
        if not tz:
            return
        try:
            pd.Timestamp.now(tz=tz)
        except Exception:
            return
        self.axis.set_timezone(tz)
        if self.case is not None:
            self.case.display["timezone"] = tz
            self.case.touch()
            self.dirty.emit()
        self.timezone_changed.emit(tz)

    def visible_bars(self) -> pd.DataFrame:
        if self.replay is None or self.raw_df.empty:
            return pd.DataFrame()
        return aggregate_visible_bars(
            self.raw_df,
            self.timeframe_combo.currentText(),
            self.replay.current_ts,
            self.replay.data_start_ts,
        )

    def render(self, reset_x: bool = False):
        if self.replay is None:
            return
        old_range = self.plot.viewRange()[0]
        self.plot.clear()
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot.addItem(self.hline, ignoreBounds=True)
        self.vline.hide(); self.hline.hide()

        bars = self.visible_bars()
        self._last_bars = bars
        if not bars.empty:
            self._draw_bars(bars)
            last = bars.iloc[-1]
            self.ohlc_label.setText(
                f"開={last['open']:.2f}   高={last['high']:.2f}   低={last['low']:.2f}   收={last['close']:.2f}"
                + ("   [forming]" if bool(last.get("is_partial", False)) else "")
            )
        else:
            self.ohlc_label.setText("開=--  高=--  低=--  收=--")

        self._render_drawing_items()

        if reset_x:
            left = self.replay.data_start_ts
            right = max(self.replay.replay_start_ts, self.replay.current_ts)
            pad = max(300, (right - left) * 0.03)
            self.plot.setXRange(left - pad, right + pad, padding=0)
            self.auto_scale()
        elif not self.auto_all_mode:
            try:
                self.plot.setXRange(old_range[0], old_range[1], padding=0)
            except Exception:
                pass

        if self.auto_all_mode:
            self.auto_all()

    def _draw_bars(self, bars: pd.DataFrame):
        t = bars["timestamp"].to_numpy(float)
        o = bars["open"].to_numpy(float)
        h = bars["high"].to_numpy(float)
        l = bars["low"].to_numpy(float)
        c = bars["close"].to_numpy(float)
        x_wick = np.repeat(t, 2)
        y_wick = np.empty(t.size * 2, dtype=float)
        y_wick[0::2] = l
        y_wick[1::2] = h
        self.plot.addItem(pg.PlotDataItem(x=x_wick, y=y_wick, connect="pairs", pen=pg.mkPen("white", width=1.1)))

        width = timeframe_seconds(self.timeframe_combo.currentText()) * 0.78
        center_y = (o + c) * 0.5
        body_h = np.maximum(np.abs(c - o), 1e-9)
        bull = c >= o
        if bull.any():
            self.plot.addItem(pg.BarGraphItem(x=t[bull], y=center_y[bull], height=body_h[bull], width=width,
                                             brush=(31, 121, 245), pen=None))
        if (~bull).any():
            self.plot.addItem(pg.BarGraphItem(x=t[~bull], y=center_y[~bull], height=body_h[~bull], width=width,
                                             brush=(247, 82, 95), pen=None))

    def auto_scale(self):
        self.auto_all_mode = False
        bars = self._last_bars
        if bars.empty:
            return
        x0, x1 = self.plot.viewRange()[0]
        sub = bars[(bars["timestamp"] >= min(x0, x1)) & (bars["timestamp"] <= max(x0, x1))]
        if sub.empty:
            sub = bars
        lo = float(sub["low"].min())
        hi = float(sub["high"].max())
        pad = max((hi - lo) * 0.08, max(abs(hi), 1.0) * 0.001)
        self.plot.setYRange(lo - pad, hi + pad, padding=0)

    def auto_all(self):
        """Fit every candle revealed by the replay and keep following new replay bars."""
        bars = self._last_bars
        if bars.empty:
            return
        self.auto_all_mode = True
        one_tick = float(timeframe_seconds(self.timeframe_combo.currentText()))
        x_min = float(bars["timestamp"].min()) - one_tick
        x_max = float(bars["timestamp"].max()) + one_tick
        if x_max <= x_min:
            x_max = x_min + max(one_tick, 1.0)
        lo = float(bars["low"].min())
        hi = float(bars["high"].max())
        pad = max((hi - lo) * 0.08, max(abs(hi), abs(lo), 1.0) * 0.002)

        self._syncing_auto_all = True
        try:
            self.plot.setXRange(x_min, x_max, padding=0.01)
            self.plot.setYRange(lo - pad, hi + pad, padding=0)
        finally:
            self._syncing_auto_all = False

    def _on_view_range_changed(self, *_args):
        # Manual pan/zoom exits persistent AutoAll mode. Internal AutoAll updates do not.
        if self.auto_all_mode and not self._syncing_auto_all:
            self.auto_all_mode = False

    def export_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "匯出圖表", "chart.png", "PNG (*.png)")
        if not path:
            return
        self.grab().save(path)

    def _mouse_moved(self, evt):
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            self.vline.hide(); self.hline.hide()
            return
        p = self.plot.vb.mapSceneToView(pos)
        self.vline.setPos(p.x()); self.hline.setPos(p.y())
        self.vline.show(); self.hline.show()

    @staticmethod
    def _is_double_click(evt) -> bool:
        try:
            return bool(evt.double())
        except Exception:
            return False

    def _drawing_id_at_scene_pos(self, scene_pos) -> str | None:
        """Return the top-most drawing id under the pointer."""
        # First use QGraphicsScene hit-testing. ROI handles are children, so walk parents.
        try:
            scene_items = self.graphics.scene().items(scene_pos)
            reverse_map = {id(item): did for did, item in self.drawing_items.items()}
            for candidate in scene_items:
                obj = candidate
                while obj is not None:
                    did = reverse_map.get(id(obj))
                    if did is not None:
                        return did
                    try:
                        obj = obj.parentItem()
                    except Exception:
                        obj = None
        except Exception:
            pass

        # Fallback to each item's shape.
        for did, item in reversed(list(self.drawing_items.items())):
            try:
                local = item.mapFromScene(scene_pos)
                if item.shape().contains(local):
                    return did
            except Exception:
                continue
        return None

    def _scene_clicked(self, evt):
        if self.case is None:
            return
        pos = evt.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return

        # Right-click always checks for a drawing first, matching the legacy UI behavior.
        if evt.button() == QtCore.Qt.RightButton:
            did = self._drawing_id_at_scene_pos(pos)
            if did is not None:
                self._show_drawing_context_menu(did)
            return

        if evt.button() != QtCore.Qt.LeftButton:
            return

        # Normal mode: only a left double-click selects a drawing. Single-click no longer selects.
        if self.tool_mode is None:
            if self._is_double_click(evt):
                did = self._drawing_id_at_scene_pos(pos)
                self._select_drawing_id(did)
            return

        p = self.plot.vb.mapSceneToView(pos)
        x, y = float(p.x()), float(p.y())

        if self.tool_mode == "horizontal_line":
            self.case.drawings.append({
                "id": f"drawing-{uuid.uuid4().hex[:12]}",
                "type": "horizontal_line",
                "price": y,
                "style": {"color": "#00ffff", "width": 2, "line_style": "solid"},
            })
            self.tool_mode = None
            self.case.touch(); self.dirty.emit(); self.rebuild_drawings(); self.render(False)
            return

        if self.tool_mode == "trend_line":
            if self.pending_point is None:
                self.pending_point = (x, y)
                return
            x1, y1 = self.pending_point
            self.case.drawings.append({
                "id": f"drawing-{uuid.uuid4().hex[:12]}",
                "type": "trend_line",
                "points": [{"time": x1, "price": y1}, {"time": x, "price": y}],
                "style": {"color": "#00ffff", "width": 2, "line_style": "solid"},
            })
            self.pending_point = None; self.tool_mode = None
            self.case.touch(); self.dirty.emit(); self.rebuild_drawings(); self.render(False)
            return

        if self.tool_mode == "text":
            text, ok = QtWidgets.QInputDialog.getMultiLineText(self, "文字註記", "內容")
            if ok and text.strip():
                self.case.drawings.append({
                    "id": f"drawing-{uuid.uuid4().hex[:12]}",
                    "type": "text",
                    "time": x,
                    "price": y,
                    "text": text.strip(),
                    "style": {"color": "#ffffff", "font_size": 12},
                })
                self.case.touch(); self.dirty.emit(); self.rebuild_drawings(); self.render(False)
            self.tool_mode = None

    def _drawing_by_id(self, did: str):
        if self.case is None:
            return None
        for drawing in self.case.drawings:
            if drawing.get("id") == did:
                return drawing
        return None

    @staticmethod
    def _qt_line_style(style_name: str):
        return {
            "solid": QtCore.Qt.SolidLine,
            "dashed": QtCore.Qt.DashLine,
            "dotted": QtCore.Qt.DotLine,
        }.get(style_name, QtCore.Qt.SolidLine)

    def _pen_from_style(self, style: dict, selected: bool = False):
        color = "#ffff00" if selected else style.get("color", "#00ffff")
        width = max(int(style.get("width", 2)) + (1 if selected else 0), 1)
        qt_style = self._qt_line_style(style.get("line_style", "solid"))
        return pg.mkPen(color, width=width, style=qt_style)

    def _select_drawing_id(self, did: str | None):
        self.selected_drawing_id = did
        self._refresh_selection_visuals()

    def _refresh_selection_visuals(self):
        if self.case is None:
            return
        by_id = {d.get("id"): d for d in self.case.drawings}
        for did, item in self.drawing_items.items():
            drawing = by_id.get(did, {})
            selected = did == self.selected_drawing_id
            try:
                item.setSelected(selected)
            except Exception:
                pass
            dtype = drawing.get("type")
            if dtype in {"horizontal_line", "trend_line"}:
                try:
                    item.setPen(self._pen_from_style(drawing.get("style", {}), selected=selected))
                except Exception:
                    pass
            elif dtype == "text":
                try:
                    base = drawing.get("style", {}).get("color", "#ffffff")
                    item.setColor("#ffff00" if selected else base)
                except Exception:
                    pass

    def _show_drawing_context_menu(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        dtype = drawing.get("type")
        menu = QtWidgets.QMenu(self)

        if dtype == "text":
            delete_action = menu.addAction("刪除")
            color_action = menu.addAction("改顏色")
            size_action = menu.addAction("改大小")
            delete_action.triggered.connect(lambda: self._delete_drawing_by_id(did))
            color_action.triggered.connect(lambda: self._change_text_color(did))
            size_action.triggered.connect(lambda: self._change_text_size(did))
        else:
            settings_action = menu.addAction("線條設定")
            delete_action = menu.addAction("刪除")
            color_action = menu.addAction("改顏色")
            settings_action.triggered.connect(lambda: self._open_line_settings(did))
            delete_action.triggered.connect(lambda: self._delete_drawing_by_id(did))
            color_action.triggered.connect(lambda: self._change_line_color(did))

        menu.exec(QtGui.QCursor.pos())

    def _open_line_settings(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style = drawing.setdefault("style", {})
        dialog = LineSettingsDialog(
            style.get("color", "#00ffff"),
            style.get("line_style", "solid"),
            int(style.get("width", 2)),
            self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        color, line_style, width = dialog.values()
        style["color"] = color
        style["line_style"] = line_style
        style["width"] = width
        self._commit_drawing_change()

    def _change_line_color(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style = drawing.setdefault("style", {})
        current = QtGui.QColor(style.get("color", "#00ffff"))
        color = QtWidgets.QColorDialog.getColor(current, self, "選擇線條顏色")
        if not color.isValid():
            return
        style["color"] = color.name()
        self._commit_drawing_change()

    def _change_text_color(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style = drawing.setdefault("style", {})
        current = QtGui.QColor(style.get("color", "#ffffff"))
        color = QtWidgets.QColorDialog.getColor(current, self, "選擇文字顏色")
        if not color.isValid():
            return
        style["color"] = color.name()
        self._commit_drawing_change()

    def _change_text_size(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style = drawing.setdefault("style", {})
        current = int(style.get("font_size", 12))
        size, ok = QtWidgets.QInputDialog.getInt(self, "字體大小", "字體大小 (pt):", current, 6, 64)
        if not ok:
            return
        style["font_size"] = int(size)
        self._commit_drawing_change()

    def _delete_drawing_by_id(self, did: str):
        if self.case is None:
            return
        self.case.drawings = [d for d in self.case.drawings if d.get("id") != did]
        if self.selected_drawing_id == did:
            self.selected_drawing_id = None
        self._commit_drawing_change()

    def _commit_drawing_change(self):
        if self.case is None:
            return
        self.case.touch()
        self.dirty.emit()
        self.render(reset_x=False)

    def rebuild_drawings(self):
        self.drawing_items.clear()
        self._render_drawing_items()

    def _render_drawing_items(self):
        if self.case is None:
            return
        self.drawing_items.clear()
        for drawing in self.case.drawings:
            dtype = drawing.get("type")
            did = drawing.get("id")
            style = drawing.get("style", {})
            if dtype == "horizontal_line":
                item = pg.InfiniteLine(
                    pos=float(drawing["price"]), angle=0, movable=True,
                    pen=self._pen_from_style(style, selected=False),
                )
                item.sigPositionChangeFinished.connect(lambda obj=item, d=drawing: self._sync_hline(obj, d))
                self.plot.addItem(item)
            elif dtype == "trend_line":
                p1, p2 = drawing["points"]
                item = pg.LineSegmentROI(
                    [(p1["time"], p1["price"]), (p2["time"], p2["price"])],
                    pen=self._pen_from_style(style, selected=False),
                )
                item.sigRegionChangeFinished.connect(lambda obj=item, d=drawing: self._sync_trend(obj, d))
                self.plot.addItem(item)
            elif dtype == "text":
                item = MovableTextItem(
                    text=drawing.get("text", ""),
                    color=style.get("color", "#ffffff"),
                    anchor=(0, 0),
                )
                font = QtGui.QFont(); font.setPointSize(int(style.get("font_size", 12)))
                try:
                    item.textItem.setFont(font)
                except Exception:
                    pass
                item.setPos(float(drawing["time"]), float(drawing["price"]))
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
                item.movementFinished.connect(lambda obj=item, d=drawing: self._sync_text(obj, d))
                self.plot.addItem(item)
            else:
                continue
            # Selection is intentionally managed by our own double-click logic.
            # Do not enable Qt's native single-click ItemIsSelectable behavior.
            try:
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
            except Exception:
                pass
            item.setToolTip(f"{dtype} / {did}")
            self.drawing_items[did] = item
        self._refresh_selection_visuals()

    def sync_all_drawings_from_view(self):
        if self.case is None:
            return
        by_id = {d.get("id"): d for d in self.case.drawings}
        for did, item in list(self.drawing_items.items()):
            d = by_id.get(did)
            if d is None:
                continue
            if d.get("type") == "horizontal_line":
                d["price"] = float(item.value())
            elif d.get("type") == "trend_line":
                self._sync_trend(item, d, emit=False)
            elif d.get("type") == "text":
                pos = item.pos()
                d["time"] = float(pos.x()); d["price"] = float(pos.y())

    def _sync_hline(self, item, drawing):
        drawing["price"] = float(item.value())
        self.case.touch(); self.dirty.emit()

    def _sync_trend(self, item, drawing, emit=True):
        try:
            pts = item.getState()["points"]
            drawing["points"] = [
                {"time": float(pts[0][0]), "price": float(pts[0][1])},
                {"time": float(pts[1][0]), "price": float(pts[1][1])},
            ]
            if emit:
                self.case.touch(); self.dirty.emit()
        except Exception:
            pass

    def _sync_text(self, item, drawing):
        try:
            pos = item.pos()
            drawing["time"] = float(pos.x())
            drawing["price"] = float(pos.y())
            self.case.touch(); self.dirty.emit()
        except Exception:
            pass

    def delete_selected_drawing(self):
        if self.case is None or self.selected_drawing_id is None:
            return
        self._delete_drawing_by_id(self.selected_drawing_id)
