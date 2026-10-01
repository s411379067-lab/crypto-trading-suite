from __future__ import annotations

import math
import uuid
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.aggregation import aggregate_visible_bars, timeframe_seconds


TRADINGVIEW_BASIC_COLORS = [
    # Red
    (242, 54, 69), (252, 203, 205), (250, 161, 164), (247, 124, 128), (247, 82, 95), (242, 54, 69),
    # Orange
    (255, 152, 0), (255, 224, 178), (255, 204, 128), (255, 183, 77), (255, 167, 38), (255, 152, 0),
    # Yellow
    (255, 235, 59), (255, 249, 196), (255, 245, 157), (255, 241, 118), (255, 238, 88), (255, 235, 59),
    # Green
    (76, 175, 80), (200, 230, 201), (165, 214, 167), (129, 199, 132), (102, 187, 106), (76, 175, 80),
    # Teal
    (8, 153, 129), (172, 229, 220), (112, 204, 189), (66, 189, 168), (34, 171, 148), (8, 153, 129),
    # Cyan
    (0, 188, 212), (178, 235, 242), (128, 222, 234), (77, 208, 225), (38, 198, 218), (0, 188, 212),
    # Blue
    (41, 98, 255), (187, 217, 251), (144, 191, 249), (91, 156, 246), (49, 121, 245), (41, 98, 255),
    # Purple
    (103, 58, 183), (209, 196, 233), (179, 157, 219), (149, 117, 205), (126, 87, 194), (103, 58, 183),
]


def get_tradingview_color(initial: str | QtGui.QColor = "#ffffff", parent=None, title: str = "選擇顏色"):
    """Legacy TradingView-style fixed color palette used by every drawing color picker."""
    dialog = QtWidgets.QColorDialog(parent)
    dialog.setWindowTitle(title)
    for i, (r, g, b) in enumerate(TRADINGVIEW_BASIC_COLORS[:48]):
        dialog.setStandardColor(i, QtGui.QColor(r, g, b))
    initial_color = initial if isinstance(initial, QtGui.QColor) else QtGui.QColor(initial)
    if initial_color.isValid():
        dialog.setCurrentColor(initial_color)
    if dialog.exec() == QtWidgets.QDialog.Accepted:
        color = dialog.currentColor()
        if color.isValid():
            return color
    return None


class TimeAxis(pg.AxisItem):
    INTERVAL_SECONDS = {
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "1H": 60 * 60,
        "4H": 4 * 60 * 60,
        "1D": 24 * 60 * 60,
    }

    def __init__(self, *args, timezone_name="Asia/Taipei", tick_interval="15m", **kwargs):
        super().__init__(*args, **kwargs)
        self.timezone_name = timezone_name
        self.tick_interval = tick_interval if tick_interval in self.INTERVAL_SECONDS else "15m"

    def set_timezone(self, timezone_name: str):
        self.timezone_name = timezone_name
        self.picture = None
        self.update()

    def set_tick_interval(self, tick_interval: str):
        if tick_interval in self.INTERVAL_SECONDS:
            self.tick_interval = tick_interval
            self.picture = None
            self.update()

    def tickValues(self, minVal, maxVal, size):
        """Only emit ticks aligned to exact local-time boundaries."""
        try:
            lo, hi = sorted((float(minVal), float(maxVal)))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return []
            step = self.INTERVAL_SECONDS[self.tick_interval]
            start_local = pd.Timestamp(lo, unit="s", tz="UTC").tz_convert(self.timezone_name)

            if self.tick_interval == "1D":
                tick_local = start_local.normalize()
                if tick_local.timestamp() < lo - 1e-9:
                    tick_local = tick_local + pd.DateOffset(days=1)
                ticks = []
                while tick_local.timestamp() <= hi + 1e-9 and len(ticks) < 5000:
                    ticks.append(float(tick_local.timestamp()))
                    tick_local = tick_local + pd.DateOffset(days=1)
            else:
                midnight = start_local.normalize()
                elapsed = (start_local - midnight).total_seconds()
                slot = int(math.ceil((elapsed - 1e-9) / step))
                tick_local = midnight + pd.Timedelta(seconds=slot * step)
                ticks = []
                while tick_local.timestamp() <= hi + 1e-9 and len(ticks) < 5000:
                    ts = float(tick_local.timestamp())
                    if ts >= lo - 1e-9:
                        ticks.append(ts)
                    tick_local = tick_local + pd.Timedelta(seconds=step)
            return [(float(step), ticks)]
        except Exception:
            return []

    def tickStrings(self, values, scale, spacing):
        labels = []
        for ts in values:
            try:
                dt = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(self.timezone_name)
                if self.tick_interval == "1D":
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
        self._color = QtGui.QColor(color if color else "#ffffff")

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
        color = get_tradingview_color(self._color, self, "選擇線條顏色")
        if color is not None and color.isValid():
            self._color = color
            self._refresh_color_button()

    def _refresh_color_button(self):
        self.color_btn.setText(self._color.name().upper())
        self.color_btn.setStyleSheet(
            f"background-color:{self._color.name()}; border:1px solid #666; color:#ffffff;"
        )

    def values(self):
        return self._color.name(), str(self.style_combo.currentData()), int(self.width_spin.value())


class RectangleSettingsDialog(QtWidgets.QDialog):
    def __init__(self, border_color: str, fill_color: str, opacity: int, width: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("長方形設定")
        self.resize(380, 245)
        self._border = QtGui.QColor(border_color or "#ffffff")
        self._fill = QtGui.QColor(fill_color or "#ffffff")

        layout = QtWidgets.QVBoxLayout(self)

        border_row = QtWidgets.QHBoxLayout()
        border_row.addWidget(QtWidgets.QLabel("外框顏色:"))
        self.border_btn = QtWidgets.QPushButton()
        self.border_btn.setFixedWidth(135)
        self.border_btn.clicked.connect(self._pick_border)
        border_row.addWidget(self.border_btn); border_row.addStretch(1)
        layout.addLayout(border_row)

        fill_row = QtWidgets.QHBoxLayout()
        fill_row.addWidget(QtWidgets.QLabel("底色:"))
        self.fill_btn = QtWidgets.QPushButton()
        self.fill_btn.setFixedWidth(135)
        self.fill_btn.clicked.connect(self._pick_fill)
        fill_row.addWidget(self.fill_btn); fill_row.addStretch(1)
        layout.addLayout(fill_row)

        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(QtWidgets.QLabel("透明度:"))
        self.opacity_spin = QtWidgets.QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setSuffix(" %")
        self.opacity_spin.setValue(max(0, min(100, int(opacity))))
        opacity_row.addWidget(self.opacity_spin); opacity_row.addStretch(1)
        layout.addLayout(opacity_row)

        width_row = QtWidgets.QHBoxLayout()
        width_row.addWidget(QtWidgets.QLabel("外框粗度:"))
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 12)
        self.width_spin.setValue(max(1, int(width)))
        width_row.addWidget(self.width_spin); width_row.addStretch(1)
        layout.addLayout(width_row)

        buttons = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("套用")
        cancel_btn = QtWidgets.QPushButton("取消")
        ok_btn.clicked.connect(self.accept); cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1); buttons.addWidget(ok_btn); buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        self._refresh_buttons()

    def _button_style(self, color: QtGui.QColor) -> str:
        brightness = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
        fg = "#000000" if brightness > 165 else "#ffffff"
        return f"background-color:{color.name()}; color:{fg}; border:1px solid #666;"

    def _refresh_buttons(self):
        self.border_btn.setText(self._border.name().upper())
        self.border_btn.setStyleSheet(self._button_style(self._border))
        self.fill_btn.setText(self._fill.name().upper())
        self.fill_btn.setStyleSheet(self._button_style(self._fill))

    def _pick_border(self):
        color = get_tradingview_color(self._border, self, "選擇外框顏色")
        if color is not None and color.isValid():
            self._border = color; self._refresh_buttons()

    def _pick_fill(self):
        color = get_tradingview_color(self._fill, self, "選擇底色")
        if color is not None and color.isValid():
            self._fill = color; self._refresh_buttons()

    def values(self):
        return self._border.name(), self._fill.name(), int(self.opacity_spin.value()), int(self.width_spin.value())


class FiboSettingsDialog(QtWidgets.QDialog):
    def __init__(self, levels: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fibo 設定")
        self.resize(430, 330)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("每列設定倍率與顏色；操作方式沿用舊版 FIBO。"))

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["倍數", "顏色"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        row_controls = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("新增列")
        remove_btn = QtWidgets.QPushButton("刪除列")
        add_btn.clicked.connect(lambda: self.add_row(1.0, "#ffffff"))
        remove_btn.clicked.connect(self.remove_selected_row)
        row_controls.addWidget(add_btn)
        row_controls.addWidget(remove_btn)
        row_controls.addStretch(1)
        layout.addLayout(row_controls)

        buttons = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("套用")
        cancel_btn = QtWidgets.QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        source = levels or [
            {"multiplier": 0.0, "color": "#ffffff"},
            {"multiplier": 0.5, "color": "#ffffff"},
            {"multiplier": 1.0, "color": "#ffffff"},
            {"multiplier": 2.0, "color": "#ffffff"},
            {"multiplier": 3.0, "color": "#ffffff"},
        ]
        for level in source:
            self.add_row(float(level.get("multiplier", 0.0)), level.get("color", "#ffffff"))

    def _style_color_button(self, button, color: str):
        qcolor = QtGui.QColor(color if color else "#ffffff")
        button.setProperty("color_hex", qcolor.name())
        button.setText(qcolor.name().upper())
        # readable text on light palette cells
        brightness = qcolor.red() * 0.299 + qcolor.green() * 0.587 + qcolor.blue() * 0.114
        fg = "#000000" if brightness > 165 else "#ffffff"
        button.setStyleSheet(f"background-color:{qcolor.name()}; color:{fg}; border:1px solid #666;")

    def _pick_color(self, button):
        current = button.property("color_hex") or "#ffffff"
        color = get_tradingview_color(current, self, "選擇 Fibo 顏色")
        if color is not None and color.isValid():
            self._style_color_button(button, color.name())

    def add_row(self, multiplier: float, color: str):
        row = self.table.rowCount()
        self.table.insertRow(row)
        item = QtWidgets.QTableWidgetItem(f"{float(multiplier):g}")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.table.setItem(row, 0, item)
        btn = QtWidgets.QPushButton()
        self._style_color_button(btn, color)
        btn.clicked.connect(lambda _=False, b=btn: self._pick_color(b))
        self.table.setCellWidget(row, 1, btn)

    def remove_selected_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def values(self) -> list[dict]:
        levels = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None or not item.text().strip():
                continue
            try:
                multiplier = float(item.text().strip())
            except Exception as exc:
                raise ValueError(f"第 {row + 1} 列倍率格式錯誤") from exc
            btn = self.table.cellWidget(row, 1)
            color = (btn.property("color_hex") if btn is not None else "#ffffff") or "#ffffff"
            levels.append({"multiplier": multiplier, "color": str(color)})
        if not levels:
            raise ValueError("請至少保留一個 Fibo level")
        return levels


class ChartWidget(QtWidgets.QWidget):
    dirty = QtCore.Signal()
    view_timeframe_changed = QtCore.Signal(str)
    timezone_changed = QtCore.Signal(str)
    order_prefill_requested = QtCore.Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_df = pd.DataFrame()
        self.replay = None
        self.case = None
        self.current_case_path = None
        self.drawing_items: dict[str, object] = {}
        self.drawing_hit_items: dict[int, str] = {}
        self._drawing_hit_objects: list[tuple[str, object]] = []
        self.tool_mode: str | None = None
        self.pending_point = None
        self._last_bars = pd.DataFrame()
        self.selected_drawing_id: str | None = None
        self.auto_all_mode = False
        self._syncing_auto_all = False
        self.order_events: list[dict] = []

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
        self.btn_rect = QtWidgets.QPushButton("□")
        self.btn_rect.setToolTip("畫長方形")
        self.btn_fibo = QtWidgets.QPushButton("Fibo")
        self.btn_shot = QtWidgets.QPushButton("Shot")
        self.btn_auto = QtWidgets.QPushButton("Auto")
        self.btn_auto_all = QtWidgets.QPushButton("AutoAll")
        for b in (self.btn_h, self.btn_l, self.btn_t, self.btn_rect):
            b.setFixedSize(40, 30)
        self.btn_fibo.setFixedSize(50, 30)
        self.btn_shot.setFixedSize(60, 30)
        self.btn_auto.setFixedSize(60, 30)
        self.btn_auto_all.setFixedSize(70, 30)

        self.timeframe_combo = QtWidgets.QComboBox()
        self.timeframe_combo.addItems(["M1", "M5", "M15", "H1"])
        self.timeframe_combo.setFixedWidth(90)

        self.x_tick_combo = QtWidgets.QComboBox()
        self.x_tick_combo.addItems(["5m", "15m", "30m", "1H", "4H", "1D"])
        self.x_tick_combo.setCurrentText("15m")
        self.x_tick_combo.setFixedWidth(75)
        self.x_tick_combo.setToolTip("X 軸時間刻度間隔；只顯示整數對齊時間")

        self.timezone_combo = QtWidgets.QComboBox()
        self.timezone_combo.setEditable(True)
        self.timezone_combo.addItems(["Asia/Taipei", "America/New_York", "UTC", "Europe/London"])
        self.timezone_combo.setFixedWidth(170)

        for w in (self.btn_h, self.btn_l, self.btn_t, self.btn_rect, self.btn_fibo, self.btn_shot, self.btn_auto, self.btn_auto_all,
                  self.timeframe_combo):
            tb.addWidget(w)
        tb.addWidget(QtWidgets.QLabel("X刻度"))
        tb.addWidget(self.x_tick_combo)
        tb.addWidget(self.timezone_combo)
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
        self.axis = TimeAxis(orientation="bottom", timezone_name="Asia/Taipei", tick_interval="15m")
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

        coord_style = (
            "background-color:#0d1420; border:1px solid #6b7fa1; border-radius:2px; "
            "color:#e6edf7; padding:2px 6px; font-size:10pt;"
        )
        self.price_coord_label = QtWidgets.QLabel(self.graphics)
        self.price_coord_label.setStyleSheet(coord_style)
        self.price_coord_label.setAlignment(QtCore.Qt.AlignCenter)
        self.price_coord_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.price_coord_label.hide()
        self.price_coord_label.raise_()

        # Legacy S/L buttons beside the Y-axis crosshair price. They only prefill the Order panel.
        self.btn_short_axis = QtWidgets.QPushButton("S", self.graphics)
        self.btn_long_axis = QtWidgets.QPushButton("L", self.graphics)
        self.btn_short_axis.setFixedSize(24, 24); self.btn_long_axis.setFixedSize(24, 24)
        self.btn_short_axis.setStyleSheet("background-color:#c41e3a; border:1px solid #8b0000; border-radius:2px; color:#fff; font-weight:bold;")
        self.btn_long_axis.setStyleSheet("background-color:#1e5a96; border:1px solid #0d3d73; border-radius:2px; color:#fff; font-weight:bold;")
        self.btn_short_axis.hide(); self.btn_long_axis.hide()
        self.btn_short_axis.raise_(); self.btn_long_axis.raise_()
        self.btn_short_axis.clicked.connect(lambda: self.order_prefill_requested.emit("short", float(self.hline.value())))
        self.btn_long_axis.clicked.connect(lambda: self.order_prefill_requested.emit("long", float(self.hline.value())))

        self.time_coord_label = QtWidgets.QLabel(self.graphics)
        self.time_coord_label.setStyleSheet(coord_style)
        self.time_coord_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_coord_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.time_coord_label.hide()
        self.time_coord_label.raise_()

        self.btn_h.clicked.connect(lambda: self.set_tool("horizontal_line"))
        self.btn_l.clicked.connect(lambda: self.set_tool("trend_line"))
        self.btn_t.clicked.connect(lambda: self.set_tool("text"))
        self.btn_rect.clicked.connect(lambda: self.set_tool("rectangle"))
        self.btn_fibo.clicked.connect(lambda: self.set_tool("fibonacci"))
        self.btn_auto.clicked.connect(self.auto_scale)
        self.btn_auto_all.clicked.connect(self.auto_all)
        self.btn_shot.clicked.connect(self.export_screenshot)
        self.timeframe_combo.currentTextChanged.connect(self._tf_changed)
        self.x_tick_combo.currentTextChanged.connect(self._x_tick_changed)
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
        self.x_tick_combo.blockSignals(True)
        self.x_tick_combo.setCurrentText(case.display.get("x_tick_interval", "15m"))
        self.x_tick_combo.blockSignals(False)
        self.timezone_combo.blockSignals(True)
        self.timezone_combo.setCurrentText(case.display.get("timezone", "Asia/Taipei"))
        self.timezone_combo.blockSignals(False)
        self.axis.set_timezone(self.timezone_combo.currentText())
        self.axis.set_tick_interval(self.x_tick_combo.currentText())

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

    def _x_tick_changed(self, interval: str):
        self.axis.set_tick_interval(interval)
        if self.case is not None:
            self.case.display["x_tick_interval"] = interval
            self.case.touch()
            self.dirty.emit()
        self.axis.picture = None
        self.axis.update()

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

    def set_order_events(self, events: list[dict], render: bool = True):
        self.order_events = list(events or [])
        if render and self.replay is not None:
            self.render(reset_x=False)

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
        self.price_coord_label.hide(); self.time_coord_label.hide()
        self.btn_short_axis.hide(); self.btn_long_axis.hide()

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
        self._render_order_markers()

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

    def _render_order_markers(self):
        if not self.order_events:
            return
        longs = [e for e in self.order_events if str(e.get("side")) == "long"]
        shorts = [e for e in self.order_events if str(e.get("side")) == "short"]
        tf_sec = float(timeframe_seconds(self.timeframe_combo.currentText()))
        def marker_x(event):
            ts = float(event["timestamp"])
            return math.floor(ts / tf_sec) * tf_sec if tf_sec > 60 else ts
        if longs:
            item = pg.ScatterPlotItem(
                x=[marker_x(e) for e in longs],
                y=[float(e["price"]) for e in longs],
                size=9, symbol="t1", brush=pg.mkBrush(0, 255, 0), pen=pg.mkPen("w", width=1),
            )
            self.plot.addItem(item)
        if shorts:
            item = pg.ScatterPlotItem(
                x=[marker_x(e) for e in shorts],
                y=[float(e["price"]) for e in shorts],
                size=9, symbol="t", brush=pg.mkBrush(255, 0, 0), pen=pg.mkPen("w", width=1),
            )
            self.plot.addItem(item)

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
            self.price_coord_label.hide(); self.time_coord_label.hide()
            self.btn_short_axis.hide(); self.btn_long_axis.hide()
            return
        p = self.plot.vb.mapSceneToView(pos)
        x = float(p.x())
        y = float(p.y())

        # TradingView-like crosshair: X snaps to the nearest revealed candle; Y follows price.
        if not self._last_bars.empty:
            ts = self._last_bars["timestamp"].to_numpy(dtype=float)
            x = float(ts[int(np.argmin(np.abs(ts - x)))])
        self.vline.setPos(x)
        self.hline.setPos(y)
        self.vline.show(); self.hline.show()
        self._update_crosshair_coordinate_labels(x, y)

    def _update_crosshair_coordinate_labels(self, x: float, y: float):
        precision = 4 if abs(y) < 10 else 2
        self.price_coord_label.setText(f"{y:.{precision}f}")
        self.price_coord_label.adjustSize()

        tz = self.timezone_combo.currentText() or "UTC"
        try:
            dt = pd.Timestamp(x, unit="s", tz="UTC").tz_convert(tz)
            self.time_coord_label.setText(dt.strftime("%Y.%m.%d  %H:%M"))
        except Exception:
            self.time_coord_label.setText("--")
        self.time_coord_label.adjustSize()

        # Y label overlays the fixed right-axis band.
        vr = self.plot.vb.viewRange()
        ref_x = (vr[0][0] + vr[0][1]) * 0.5
        scene_y = self.plot.vb.mapViewToScene(QtCore.QPointF(ref_x, y))
        widget_y = self.graphics.mapFromScene(scene_y)
        pw, ph = self.price_coord_label.width(), self.price_coord_label.height()
        px = max(0, self.graphics.width() - 72 + 2)
        py = max(0, min(int(widget_y.y() - ph * 0.5), self.graphics.height() - ph))
        self.price_coord_label.move(px, py)
        self.price_coord_label.show()

        btn_size = 24
        btn_spacing = 2
        total_btn_w = btn_size * 2 + btn_spacing
        btn_x = max(0, px - total_btn_w - 3)
        btn_y = max(0, min(int(py + (ph - btn_size) * 0.5), self.graphics.height() - btn_size))
        self.btn_short_axis.move(btn_x, btn_y)
        self.btn_long_axis.move(btn_x + btn_size + btn_spacing, btn_y)
        self.btn_short_axis.show(); self.btn_long_axis.show()

        # X label sits on the bottom axis and follows the snapped candle time.
        scene_x = self.plot.vb.mapViewToScene(QtCore.QPointF(x, vr[1][0]))
        widget_x = self.graphics.mapFromScene(scene_x)
        tw, th = self.time_coord_label.width(), self.time_coord_label.height()
        tx = max(0, min(int(widget_x.x() - tw * 0.5), self.graphics.width() - tw))
        ty = max(0, self.graphics.height() - th - 2)
        self.time_coord_label.move(tx, ty)
        self.time_coord_label.show()
        self.price_coord_label.raise_(); self.time_coord_label.raise_(); self.btn_short_axis.raise_(); self.btn_long_axis.raise_()

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
            reverse_map = self.drawing_hit_items
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

        # Fallback to each registered graphics item's shape.
        for did, item in reversed(self._drawing_hit_objects):
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
                "style": {"color": "#ffffff", "width": 2, "line_style": "solid"},
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
                "style": {"color": "#ffffff", "width": 2, "line_style": "solid"},
            })
            self.pending_point = None; self.tool_mode = None
            self.case.touch(); self.dirty.emit(); self.rebuild_drawings(); self.render(False)
            return

        if self.tool_mode == "rectangle":
            if self.pending_point is None:
                self.pending_point = (x, y)
                return
            x1, y1 = self.pending_point
            self.case.drawings.append({
                "id": f"drawing-{uuid.uuid4().hex[:12]}",
                "type": "rectangle",
                "points": [{"time": x1, "price": y1}, {"time": x, "price": y}],
                "style": {"border_color": "#ffffff", "fill_color": "#ffffff", "opacity": 12, "width": 2},
            })
            self.pending_point = None; self.tool_mode = None
            self.case.touch(); self.dirty.emit(); self.rebuild_drawings(); self.render(False)
            return

        if self.tool_mode == "fibonacci":
            if self.pending_point is None:
                self.pending_point = (x, y)
                return
            x1, y1 = self.pending_point
            self.case.drawings.append({
                "id": f"drawing-{uuid.uuid4().hex[:12]}",
                "type": "fibonacci",
                "start": {"time": x1, "price": y1},
                "end": {"time": x, "price": y},
                "levels": [
                    {"multiplier": 0.0, "color": "#ffffff"},
                    {"multiplier": 0.5, "color": "#ffffff"},
                    {"multiplier": 1.0, "color": "#ffffff"},
                    {"multiplier": 2.0, "color": "#ffffff"},
                    {"multiplier": 3.0, "color": "#ffffff"},
                ],
                "style": {"width": 2, "line_style": "solid"},
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
        color = "#ffff00" if selected else style.get("color", "#ffffff")
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
            elif dtype == "rectangle" and isinstance(item, dict):
                roi = item.get("roi")
                if roi is not None:
                    style = drawing.get("style", {})
                    base = style.get("border_color", "#ffffff")
                    width = int(style.get("width", 2))
                    roi.setPen(pg.mkPen("#ffff00" if selected else base, width=width + (1 if selected else 0)))
            elif dtype == "fibonacci" and isinstance(item, dict):
                levels = drawing.get("levels", [])
                width = int(drawing.get("style", {}).get("width", 2))
                for i, line in enumerate(item.get("lines", [])):
                    base = levels[i].get("color", "#ffffff") if i < len(levels) else "#ffffff"
                    line.setPen(pg.mkPen("#ffff00" if selected else base, width=width + (1 if selected else 0)))
                try:
                    item["box"].setPen(pg.mkPen((255, 255, 0, 120) if selected else (0, 0, 0, 0), width=1))
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
        elif dtype == "rectangle":
            settings_action = menu.addAction("長方形設定")
            delete_action = menu.addAction("刪除")
            settings_action.triggered.connect(lambda: self._open_rectangle_settings(did))
            delete_action.triggered.connect(lambda: self._delete_drawing_by_id(did))
        elif dtype == "fibonacci":
            settings_action = menu.addAction("Fibo 設定")
            delete_action = menu.addAction("刪除")
            color_action = menu.addAction("改顏色")
            settings_action.triggered.connect(lambda: self._open_fibo_settings(did))
            delete_action.triggered.connect(lambda: self._delete_drawing_by_id(did))
            color_action.triggered.connect(lambda: self._change_fibo_color(did))
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
            style.get("color", "#ffffff"),
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
        current = QtGui.QColor(style.get("color", "#ffffff"))
        color = get_tradingview_color(current, self, "選擇線條顏色")
        if color is None or not color.isValid():
            return
        style["color"] = color.name()
        self._commit_drawing_change()

    def _change_text_color(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style = drawing.setdefault("style", {})
        current = QtGui.QColor(style.get("color", "#ffffff"))
        color = get_tradingview_color(current, self, "選擇文字顏色")
        if color is None or not color.isValid():
            return
        style["color"] = color.name()
        self._commit_drawing_change()

    def _open_rectangle_settings(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style = drawing.setdefault("style", {})
        dialog = RectangleSettingsDialog(
            style.get("border_color", "#ffffff"),
            style.get("fill_color", "#ffffff"),
            int(style.get("opacity", 12)),
            int(style.get("width", 2)),
            self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        border, fill, opacity, width = dialog.values()
        style["border_color"] = border
        style["fill_color"] = fill
        style["opacity"] = opacity
        style["width"] = width
        self._commit_drawing_change()

    def _open_fibo_settings(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        dialog = FiboSettingsDialog(drawing.get("levels", []), self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            drawing["levels"] = dialog.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Fibo 設定", str(exc))
            return
        self._commit_drawing_change()

    def _change_fibo_color(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        levels = drawing.setdefault("levels", [])
        current = levels[0].get("color", "#ffffff") if levels else "#ffffff"
        color = get_tradingview_color(current, self, "選擇 Fibo 顏色")
        if color is None or not color.isValid():
            return
        if not levels:
            levels.extend([
                {"multiplier": 0.0, "color": color.name()},
                {"multiplier": 0.5, "color": color.name()},
                {"multiplier": 1.0, "color": color.name()},
                {"multiplier": 2.0, "color": color.name()},
                {"multiplier": 3.0, "color": color.name()},
            ])
        else:
            for level in levels:
                level["color"] = color.name()
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
        self.drawing_hit_items.clear()
        self._drawing_hit_objects.clear()
        self._render_drawing_items()

    def _register_drawing_hit_item(self, did: str, item):
        self.drawing_hit_items[id(item)] = did
        self._drawing_hit_objects.append((did, item))

    @staticmethod
    def _fibo_line_y(drawing: dict, multiplier: float) -> float:
        start_y = float(drawing["start"]["price"])
        end_y = float(drawing["end"]["price"])
        return start_y + (end_y - start_y) * float(multiplier)

    def _fibo_geometry(self, drawing: dict):
        sx = float(drawing["start"]["time"]); sy = float(drawing["start"]["price"])
        ex = float(drawing["end"]["time"]); ey = float(drawing["end"]["price"])
        levels = drawing.get("levels", []) or [{"multiplier": 0.0, "color": "#ffffff"}]
        ys = [self._fibo_line_y(drawing, lv.get("multiplier", 0.0)) for lv in levels]
        left, right = min(sx, ex), max(sx, ex)
        if right <= left:
            right = left + 1e-6
        ymin, ymax = min(ys), max(ys)
        if ymax <= ymin:
            ymax = ymin + 1e-6
        return sx, sy, ex, ey, left, right, ymin, ymax, ys

    def _set_rect_line(self, rect, left: float, right: float, y: float, height: float):
        rect.blockSignals(True)
        try:
            rect.setPos([left, y - height * 0.5])
            rect.setSize([max(right - left, 1e-6), max(height, 1e-6)])
        finally:
            rect.blockSignals(False)

    def _update_fibo_view_geometry(self, did: str, update_box: bool = True):
        drawing = self._drawing_by_id(did)
        group = self.drawing_items.get(did)
        if drawing is None or not isinstance(group, dict):
            return
        sx, sy, ex, ey, left, right, ymin, ymax, ys = self._fibo_geometry(drawing)
        tick = max(float(timeframe_seconds(self.timeframe_combo.currentText())), 1.0)
        h = max((ymax - ymin) * 0.003, max(abs(ymax), 1.0) * 1e-6)
        handle_left, handle_right = left - 0.5 * tick, right + 0.5 * tick
        if update_box:
            box = group.get("box")
            if box is not None:
                box.blockSignals(True)
                try:
                    box.setPos([left, ymin]); box.setSize([right - left, ymax - ymin])
                finally:
                    box.blockSignals(False)
        self._set_rect_line(group["start_handle"], handle_left, handle_right, sy, h)
        self._set_rect_line(group["end_handle"], handle_left, handle_right, ey, h)
        group["handle_height"] = h
        group["last_box_rect"] = [left, right, ymin, ymax]
        width = int(drawing.get("style", {}).get("width", 2))
        selected = did == self.selected_drawing_id
        levels = drawing.get("levels", [])
        for i, (line, y) in enumerate(zip(group.get("lines", []), ys)):
            line.setData([sx, ex], [y, y])
            base = levels[i].get("color", "#ffffff") if i < len(levels) else "#ffffff"
            line.setPen(pg.mkPen("#ffff00" if selected else base, width=width + (1 if selected else 0)))

    def _sync_fibo_from_box(self, did: str):
        drawing = self._drawing_by_id(did)
        group = self.drawing_items.get(did)
        if drawing is None or not isinstance(group, dict) or group.get("syncing"):
            return
        box = group.get("box")
        if box is None:
            return
        group["syncing"] = True
        try:
            pos = box.pos(); size = box.size()
            curr_left = float(pos.x()); curr_right = float(pos.x() + size.x())
            curr_ymin = float(pos.y()); curr_ymax = float(pos.y() + size.y())
            last_left, last_right, last_ymin, last_ymax = group.get(
                "last_box_rect", [curr_left, curr_right, curr_ymin, curr_ymax]
            )
            start_is_left = float(drawing["start"]["time"]) <= float(drawing["end"]["time"])
            drawing["start"]["time"] = curr_left if start_is_left else curr_right
            drawing["end"]["time"] = curr_right if start_is_left else curr_left
            dy = curr_ymin - float(last_ymin)
            drawing["start"]["price"] = float(drawing["start"]["price"]) + dy
            drawing["end"]["price"] = float(drawing["end"]["price"]) + dy
            self._update_fibo_view_geometry(did, update_box=True)
            self.case.touch(); self.dirty.emit()
        finally:
            group["syncing"] = False

    def _sync_fibo_from_handles(self, did: str):
        drawing = self._drawing_by_id(did)
        group = self.drawing_items.get(did)
        if drawing is None or not isinstance(group, dict) or group.get("syncing"):
            return
        group["syncing"] = True
        try:
            sh = group.get("start_handle"); eh = group.get("end_handle")
            sp, ss = sh.pos(), sh.size(); ep, es = eh.pos(), eh.size()
            drawing["start"]["price"] = float(sp.y() + ss.y() * 0.5)
            drawing["end"]["price"] = float(ep.y() + es.y() * 0.5)
            self._update_fibo_view_geometry(did, update_box=True)
            self.case.touch(); self.dirty.emit()
        finally:
            group["syncing"] = False

    def _render_fibonacci(self, drawing: dict):
        did = drawing.get("id")
        sx, sy, ex, ey, left, right, ymin, ymax, ys = self._fibo_geometry(drawing)
        tick = max(float(timeframe_seconds(self.timeframe_combo.currentText())), 1.0)
        handle_h = max((ymax - ymin) * 0.003, max(abs(ymax), 1.0) * 1e-6)
        handle_left, handle_right = left - 0.5 * tick, right + 0.5 * tick

        box = pg.RectROI([left, ymin], [right - left, ymax - ymin], pen=pg.mkPen((0, 0, 0, 0)), movable=True)
        try:
            box.setHoverPen(pg.mkPen((255, 255, 0, 100), width=1))
        except Exception:
            pass
        self.plot.addItem(box)

        start_handle = pg.RectROI(
            [handle_left, sy - handle_h * 0.5], [handle_right - handle_left, handle_h],
            pen=pg.mkPen((255, 80, 80), width=2), movable=True, rotatable=False, resizable=False,
        )
        end_handle = pg.RectROI(
            [handle_left, ey - handle_h * 0.5], [handle_right - handle_left, handle_h],
            pen=pg.mkPen((255, 80, 80), width=2), movable=True, rotatable=False, resizable=False,
        )
        start_handle.setZValue(9); end_handle.setZValue(9)
        self.plot.addItem(start_handle); self.plot.addItem(end_handle)

        lines = []
        levels = drawing.get("levels", [])
        width = int(drawing.get("style", {}).get("width", 2))
        for i, y in enumerate(ys):
            color = levels[i].get("color", "#ffffff") if i < len(levels) else "#ffffff"
            line = pg.PlotDataItem(x=[sx, ex], y=[y, y], pen=pg.mkPen(color, width=width))
            line.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self.plot.addItem(line)
            lines.append(line)

        group = {
            "type": "fibonacci", "box": box, "start_handle": start_handle, "end_handle": end_handle,
            "lines": lines, "last_box_rect": [left, right, ymin, ymax], "syncing": False,
            "handle_height": handle_h,
        }
        self.drawing_items[did] = group
        for item in [box, start_handle, end_handle, *lines]:
            self._register_drawing_hit_item(did, item)
            try:
                item.setToolTip(f"fibonacci / {did}")
            except Exception:
                pass
        box.sigRegionChanged.connect(lambda _=None, d=did: self._sync_fibo_from_box(d))
        start_handle.sigRegionChanged.connect(lambda _=None, d=did: self._sync_fibo_from_handles(d))
        end_handle.sigRegionChanged.connect(lambda _=None, d=did: self._sync_fibo_from_handles(d))

    def _rectangle_geometry(self, drawing: dict):
        p1, p2 = drawing["points"]
        x1, y1 = float(p1["time"]), float(p1["price"])
        x2, y2 = float(p2["time"]), float(p2["price"])
        left, right = min(x1, x2), max(x1, x2)
        bottom, top = min(y1, y2), max(y1, y2)
        if right <= left:
            right = left + 1e-6
        if top <= bottom:
            top = bottom + 1e-9
        return left, right, bottom, top

    def _update_rectangle_fill(self, group: dict):
        roi = group.get("roi"); fill = group.get("fill")
        if roi is None or fill is None:
            return
        try:
            pos = roi.pos(); size = roi.size()
            fill.setRect(float(pos.x()), float(pos.y()), float(size.x()), float(size.y()))
        except Exception:
            pass

    def _render_rectangle(self, drawing: dict):
        did = drawing.get("id")
        left, right, bottom, top = self._rectangle_geometry(drawing)
        style = drawing.get("style", {})
        border = style.get("border_color", "#ffffff")
        fill_color = QtGui.QColor(style.get("fill_color", "#ffffff"))
        alpha = int(round(max(0, min(100, int(style.get("opacity", 12)))) * 255 / 100.0))
        fill_color.setAlpha(alpha)

        fill_item = QtWidgets.QGraphicsRectItem(left, bottom, right - left, top - bottom)
        fill_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        fill_item.setBrush(QtGui.QBrush(fill_color))
        fill_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        fill_item.setZValue(3)
        self.plot.addItem(fill_item)

        roi = pg.RectROI(
            [left, bottom], [right - left, top - bottom],
            pen=pg.mkPen(border, width=int(style.get("width", 2))), movable=True,
        )
        roi.setZValue(4)
        self.plot.addItem(roi)
        group = {"type": "rectangle", "roi": roi, "fill": fill_item}
        self.drawing_items[did] = group
        self._register_drawing_hit_item(did, roi)
        self._register_drawing_hit_item(did, fill_item)
        roi.sigRegionChanged.connect(lambda _=None, d=did: self._update_rectangle_fill(self.drawing_items.get(d, {})))
        roi.sigRegionChangeFinished.connect(lambda obj=roi, d=drawing: self._sync_rectangle(obj, d))

    def _render_drawing_items(self):
        if self.case is None:
            return
        self.drawing_items.clear()
        self.drawing_hit_items.clear()
        self._drawing_hit_objects.clear()
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
            elif dtype == "rectangle":
                self._render_rectangle(drawing)
                continue
            elif dtype == "fibonacci":
                self._render_fibonacci(drawing)
                continue
            else:
                continue
            try:
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
            except Exception:
                pass
            item.setToolTip(f"{dtype} / {did}")
            self.drawing_items[did] = item
            self._register_drawing_hit_item(did, item)
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
            elif d.get("type") == "rectangle":
                roi = item.get("roi") if isinstance(item, dict) else None
                if roi is not None:
                    self._sync_rectangle(roi, d, emit=False)
            elif d.get("type") == "fibonacci":
                # Fibo domain data is updated live from the box/0x/1x handles.
                pass

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

    def _sync_rectangle(self, item, drawing, emit=True):
        try:
            pos = item.pos(); size = item.size()
            left = float(pos.x()); bottom = float(pos.y())
            right = left + float(size.x()); top = bottom + float(size.y())
            drawing["points"] = [
                {"time": left, "price": bottom},
                {"time": right, "price": top},
            ]
            group = self.drawing_items.get(drawing.get("id"))
            if isinstance(group, dict):
                self._update_rectangle_fill(group)
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
