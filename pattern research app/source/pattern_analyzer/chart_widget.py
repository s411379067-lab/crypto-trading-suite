from __future__ import annotations

import math
import uuid
from copy import deepcopy
from typing import Callable
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.geometry_picker import point_to_rect_distance, point_to_segment_distance
from shared_core.aggregation import aggregate_visible_bars, timeframe_seconds
from shared_core.drawing_clipboard import clone_drawing_with_offset
from shared_core.note_callout import align_note_x_to_timeframe, find_m1_close, layout_alternating_note_callouts, resolve_note_timestamp, wrap_note_text
from shared_core.rth import summarize_intraday_volatility_payload
from shared_core.replay_stats import summarize_current_replay_range
from shared_core.ohlc import format_ohlc_text
from shared_core.order_brackets import (
    available_bracket_qty,
    move_bracket_entry,
    move_bracket_leg,
    set_bracket_group_qty,
    set_bracket_qty,
)
from pattern_analyzer.drawing_templates import DrawingTemplateRepository


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
    """Text item with drag persistence, Ctrl magnet snapping, and a reliable custom context menu."""
    movementFinished = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magnet_callback = None
        self.context_menu_callback = None

        # Keep left-button dragging on the TextItem itself, but prevent the internal
        # QGraphicsTextItem from swallowing right-clicks.  This was the reason text
        # drawings could not open the common drawing context menu in v0.1.8.
        try:
            self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.RightButton)
            self.textItem.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self.textItem.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        except Exception:
            pass

    def contextMenuEvent(self, ev):
        if callable(self.context_menu_callback):
            self.context_menu_callback()
            ev.accept()
            return
        try:
            super().contextMenuEvent(ev)
        except Exception:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        try:
            ctrl = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier)
            if ctrl and callable(self.magnet_callback):
                pos = self.pos()
                sx, sy = self.magnet_callback(float(pos.x()), float(pos.y()))
                self.setPos(float(sx), float(sy))
        except Exception:
            pass

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.movementFinished.emit()


class LineSettingsDialog(QtWidgets.QDialog):
    STYLE_OPTIONS = [
        ("實線", "solid"),
        ("虛線", "dashed"),
        ("點線", "dotted"),
    ]

    def __init__(self, color: str, style_name: str, width_value: int,
                 save_template_callback: Callable[[dict], None] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("線條設定")
        self.resize(365, 220)
        self._color = QtGui.QColor(color if color else "#ffffff")
        self._save_template_callback = save_template_callback

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
        save_btn = QtWidgets.QPushButton("存為模板...")
        save_btn.setEnabled(self._save_template_callback is not None)
        save_btn.clicked.connect(self._save_template)
        ok_btn = QtWidgets.QPushButton("套用")
        cancel_btn = QtWidgets.QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(save_btn)
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
        brightness = self._color.red() * 0.299 + self._color.green() * 0.587 + self._color.blue() * 0.114
        fg = "#000000" if brightness > 165 else "#ffffff"
        self.color_btn.setStyleSheet(
            f"background-color:{self._color.name()}; border:1px solid #666; color:{fg};"
        )

    def values(self):
        return self._color.name(), str(self.style_combo.currentData()), int(self.width_spin.value())

    def _save_template(self):
        if self._save_template_callback is None:
            return
        color, line_style, width = self.values()
        self._save_template_callback({"style": {"color": color, "line_style": line_style, "width": width}})


class RectangleSettingsDialog(QtWidgets.QDialog):
    def __init__(self, border_color: str, fill_color: str, opacity: int, width: int,
                 save_template_callback: Callable[[dict], None] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("長方形設定")
        self.resize(395, 265)
        self._border = QtGui.QColor(border_color or "#ffffff")
        self._fill = QtGui.QColor(fill_color or "#ffffff")
        self._save_template_callback = save_template_callback

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
        save_btn = QtWidgets.QPushButton("存為模板...")
        save_btn.setEnabled(self._save_template_callback is not None)
        save_btn.clicked.connect(self._save_template)
        ok_btn = QtWidgets.QPushButton("套用")
        cancel_btn = QtWidgets.QPushButton("取消")
        ok_btn.clicked.connect(self.accept); cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(save_btn)
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

    def _save_template(self):
        if self._save_template_callback is None:
            return
        border, fill, opacity, width = self.values()
        self._save_template_callback({
            "style": {
                "border_color": border,
                "fill_color": fill,
                "opacity": opacity,
                "width": width,
            }
        })


class TextSettingsDialog(QtWidgets.QDialog):
    """TradingView-like text box editor.

    Text content and drawing style are edited together. Templates save style only;
    geometry and text content remain owned by the Case JSON.
    """
    def __init__(
        self,
        text: str,
        text_color: str,
        font_size: int,
        bold: bool = False,
        italic: bool = False,
        border_enabled: bool = False,
        border_color: str = "#ffffff",
        border_width: int = 1,
        background_enabled: bool = False,
        background_color: str = "#181c27",
        background_opacity: int = 0,
        auto_wrap: bool = True,
        save_template_callback: Callable[[dict], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("文字設定")
        self.resize(500, 520)
        self._text_color = QtGui.QColor(text_color or "#ffffff")
        self._border_color = QtGui.QColor(border_color or "#ffffff")
        self._background_color = QtGui.QColor(background_color or "#181c27")
        self._save_template_callback = save_template_callback

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # TradingView-like compact text toolbar.
        toolbar = QtWidgets.QHBoxLayout()
        self.text_color_btn = QtWidgets.QPushButton()
        self.text_color_btn.setFixedSize(42, 34)
        self.text_color_btn.setToolTip("文字顏色")
        self.text_color_btn.clicked.connect(self._pick_text_color)
        toolbar.addWidget(self.text_color_btn)

        self.size_combo = QtWidgets.QComboBox()
        self.size_combo.addItems(["8", "9", "10", "11", "12", "14", "16", "18", "20", "24", "28", "32", "36", "48", "64"])
        if self.size_combo.findText(str(int(font_size))) < 0:
            self.size_combo.addItem(str(int(font_size)))
        self.size_combo.setCurrentText(str(int(font_size)))
        self.size_combo.setFixedWidth(92)
        toolbar.addWidget(self.size_combo)

        toggle_style = (
            "QPushButton { background-color:#1f1f1f; color:#e8edf6; border:1px solid #4b5260; "
            "border-radius:5px; }"
            "QPushButton:hover { border:1px solid #7a8598; }"
            "QPushButton:checked { background-color:#d7d7d7; color:#111111; "
            "border:1px solid #f3f3f3; }"
        )

        self.bold_btn = QtWidgets.QPushButton("B")
        self.bold_btn.setCheckable(True)
        self.bold_btn.setChecked(bool(bold))
        self.bold_btn.setFixedSize(44, 34)
        self.bold_btn.setStyleSheet(toggle_style)
        bold_font = self.bold_btn.font(); bold_font.setBold(True); self.bold_btn.setFont(bold_font)
        toolbar.addWidget(self.bold_btn)

        self.italic_btn = QtWidgets.QPushButton("I")
        self.italic_btn.setCheckable(True)
        self.italic_btn.setChecked(bool(italic))
        self.italic_btn.setFixedSize(44, 34)
        self.italic_btn.setStyleSheet(toggle_style)
        italic_font = self.italic_btn.font(); italic_font.setItalic(True); self.italic_btn.setFont(italic_font)
        toolbar.addWidget(self.italic_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setPlainText(text or "")
        self.text_edit.setMinimumHeight(205)
        self.text_edit.setPlaceholderText("輸入文字...")
        layout.addWidget(self.text_edit)

        # Background controls are included because they are part of the TradingView text-box workflow.
        bg_row = QtWidgets.QHBoxLayout()
        self.background_check = QtWidgets.QCheckBox("背景")
        self.background_check.setChecked(bool(background_enabled))
        bg_row.addWidget(self.background_check)
        self.background_color_btn = QtWidgets.QPushButton()
        self.background_color_btn.setFixedSize(44, 32)
        self.background_color_btn.clicked.connect(self._pick_background_color)
        bg_row.addWidget(self.background_color_btn)
        bg_row.addWidget(QtWidgets.QLabel("透明度"))
        self.background_opacity_spin = QtWidgets.QSpinBox()
        self.background_opacity_spin.setRange(0, 100)
        self.background_opacity_spin.setSuffix(" %")
        self.background_opacity_spin.setValue(max(0, min(100, int(background_opacity))))
        bg_row.addWidget(self.background_opacity_spin)
        bg_row.addStretch(1)
        layout.addLayout(bg_row)

        border_row = QtWidgets.QHBoxLayout()
        self.border_check = QtWidgets.QCheckBox("框線")
        self.border_check.setChecked(bool(border_enabled))
        border_row.addWidget(self.border_check)
        self.border_color_btn = QtWidgets.QPushButton()
        self.border_color_btn.setFixedSize(44, 32)
        self.border_color_btn.clicked.connect(self._pick_border_color)
        border_row.addWidget(self.border_color_btn)
        border_row.addWidget(QtWidgets.QLabel("粗度"))
        self.border_width_spin = QtWidgets.QSpinBox()
        self.border_width_spin.setRange(1, 12)
        self.border_width_spin.setValue(max(1, int(border_width or 1)))
        border_row.addWidget(self.border_width_spin)
        border_row.addStretch(1)
        layout.addLayout(border_row)

        self.wrap_check = QtWidgets.QCheckBox("自動換行")
        self.wrap_check.setChecked(bool(auto_wrap))
        layout.addWidget(self.wrap_check)

        buttons = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("存為模板...")
        save_btn.setEnabled(self._save_template_callback is not None)
        save_btn.clicked.connect(self._save_template)
        ok_btn = QtWidgets.QPushButton("套用")
        cancel_btn = QtWidgets.QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(save_btn)
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        self._refresh_color_buttons()

    @staticmethod
    def _button_style(color: QtGui.QColor) -> str:
        brightness = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
        fg = "#000000" if brightness > 165 else "#ffffff"
        return f"background-color:{color.name()}; color:{fg}; border:1px solid #666; border-radius:4px;"

    def _refresh_color_buttons(self):
        for button, color in (
            (self.text_color_btn, self._text_color),
            (self.border_color_btn, self._border_color),
            (self.background_color_btn, self._background_color),
        ):
            button.setText("")
            button.setStyleSheet(self._button_style(color))

    def _pick_text_color(self):
        color = get_tradingview_color(self._text_color, self, "選擇文字顏色")
        if color is not None and color.isValid():
            self._text_color = color
            self._refresh_color_buttons()

    def _pick_border_color(self):
        color = get_tradingview_color(self._border_color, self, "選擇框線顏色")
        if color is not None and color.isValid():
            self._border_color = color
            self._refresh_color_buttons()

    def _pick_background_color(self):
        color = get_tradingview_color(self._background_color, self, "選擇背景顏色")
        if color is not None and color.isValid():
            self._background_color = color
            self._refresh_color_buttons()

    def values(self):
        style = {
            "color": self._text_color.name(),
            "font_size": int(self.size_combo.currentText()),
            "bold": bool(self.bold_btn.isChecked()),
            "italic": bool(self.italic_btn.isChecked()),
            "border_enabled": bool(self.border_check.isChecked()),
            "border_color": self._border_color.name(),
            "border_width": int(self.border_width_spin.value()),
            "background_enabled": bool(self.background_check.isChecked()),
            "background_color": self._background_color.name(),
            "background_opacity": int(self.background_opacity_spin.value()),
            "auto_wrap": bool(self.wrap_check.isChecked()),
        }
        return self.text_edit.toPlainText(), style

    def _save_template(self):
        if self._save_template_callback is None:
            return
        _text, style = self.values()
        self._save_template_callback({"style": style})


class FiboSettingsDialog(QtWidgets.QDialog):
    STYLE_OPTIONS = [
        ("實線", "solid"),
        ("虛線", "dashed"),
        ("點線", "dotted"),
    ]

    def __init__(self, levels: list[dict], style: dict | None = None,
                 save_template_callback: Callable[[dict], None] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fibo 設定")
        self.resize(465, 430)
        self._save_template_callback = save_template_callback
        self._style = deepcopy(style or {"width": 2, "line_style": "solid"})
        layout = QtWidgets.QVBoxLayout(self)

        style_box = QtWidgets.QGroupBox("整體線條")
        style_layout = QtWidgets.QHBoxLayout(style_box)
        style_layout.addWidget(QtWidgets.QLabel("線型:"))
        self.style_combo = QtWidgets.QComboBox()
        for label, key in self.STYLE_OPTIONS:
            self.style_combo.addItem(label, key)
        idx = self.style_combo.findData(str(self._style.get("line_style", "solid")))
        self.style_combo.setCurrentIndex(idx if idx >= 0 else 0)
        style_layout.addWidget(self.style_combo)
        style_layout.addSpacing(14)
        style_layout.addWidget(QtWidgets.QLabel("線粗:"))
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 12)
        self.width_spin.setValue(max(1, int(self._style.get("width", 2))))
        style_layout.addWidget(self.width_spin)
        style_layout.addStretch(1)
        layout.addWidget(style_box)

        layout.addWidget(QtWidgets.QLabel("每列設定倍率與顏色。選取 Fibo 後，0 / 1 兩個錨點可自由調整時間與價格。"))

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
        save_btn = QtWidgets.QPushButton("存為模板...")
        save_btn.setEnabled(self._save_template_callback is not None)
        save_btn.clicked.connect(self._save_template)
        ok_btn = QtWidgets.QPushButton("套用")
        cancel_btn = QtWidgets.QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(save_btn)
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

    def _level_values(self) -> list[dict]:
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

    def values(self) -> tuple[list[dict], dict]:
        levels = self._level_values()
        style = deepcopy(self._style)
        style["line_style"] = str(self.style_combo.currentData() or "solid")
        style["width"] = int(self.width_spin.value())
        return levels, style

    def _save_template(self):
        if self._save_template_callback is None:
            return
        try:
            levels, style = self.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Fibo 設定", str(exc))
            return
        self._save_template_callback({"levels": levels, "style": style})


class ChartWidget(QtWidgets.QWidget):
    dirty = QtCore.Signal()
    history_committed = QtCore.Signal(str)
    view_timeframe_changed = QtCore.Signal(str)
    timezone_changed = QtCore.Signal(str)
    order_prefill_requested = QtCore.Signal(str, float)
    pending_bracket_edited = QtCore.Signal(object)

    def __init__(self, parent=None, *, drawing_interaction_enabled: bool = True):
        super().__init__(parent)
        # Analyzer uses the full editable Drawing system. Viewer passes False so
        # Drawings are created as non-interactive view objects from the outset,
        # rather than being editable ROI objects that are locked after render.
        self.drawing_interaction_enabled = bool(drawing_interaction_enabled)
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
        self._drawing_clipboard: dict | None = None
        self._clipboard_paste_count = 0
        self.template_repo = DrawingTemplateRepository()
        self.auto_all_mode = False
        self._syncing_auto_all = False
        self.order_events: list[dict] = []
        self.order_segments: list[dict] = []
        self.pending_bracket: dict | None = None
        self._pending_qty_controls: list[tuple[QtWidgets.QWidget, float]] = []
        self.show_all_orders = False
        self.show_previous_rth = False
        self.show_all_drawings = True
        self.show_all_intraday_notes = False
        self.selected_intraday_note: dict | None = None
        self._all_note_callout_items: list[object] = []

        # Transient measure mode. Nothing here is persisted to Case JSON.
        self.measure_mode = False
        self._measure_dragging = False
        self._measure_start: tuple[float, float] | None = None
        self._measure_end: tuple[float, float] | None = None

        pg.setConfigOption("background", "#181c27")
        pg.setConfigOption("foreground", "white")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

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
        self.show_drawings_checkbox = QtWidgets.QCheckBox("顯示全部圖形")
        self.show_drawings_checkbox.setChecked(True)
        self.show_drawings_checkbox.setToolTip("一次顯示 / 隱藏所有 Drawing；不刪除也不修改 Case JSON")
        self.show_orders_checkbox = QtWidgets.QCheckBox("顯示全部 Order")
        self.show_orders_checkbox.setChecked(False)
        self.show_orders_checkbox.setToolTip("顯示已揭露成交的紅綠三角形，以及已完成交易的 Open → Close PnL 點線；不修改 Case JSON")
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

        self.vol_n_label = QtWidgets.QLabel("波動N")
        self.vol_n_spin = QtWidgets.QSpinBox()
        self.vol_n_spin.setRange(2, 20)
        self.vol_n_spin.setValue(20)
        self.vol_n_spin.setFixedWidth(58)
        self.vol_n_spin.setToolTip("使用 Enricher 已寫入 Case JSON 的最近 N 個完整 RTH Session；不重新讀取 raw data")
        self.vol_stats_label = QtWidgets.QLabel("Med -- | σ -- | 2σ -- | 3σ --")
        self.vol_stats_label.setStyleSheet("color:#d7deea; font-weight:600; padding-left:4px;")
        self.vol_stats_label.setToolTip("Range % = (RTH High - RTH Low) / RTH Open × 100；σ 使用樣本標準差 ddof=1")
        self.current_range_label = QtWidgets.QLabel("目前 Range --")
        self.current_range_label.setStyleSheet("color:#ffcc80; font-weight:700; padding-left:8px;")
        self.current_range_label.setToolTip(
            "從 replay_start 到目前 Replay 時間，使用已揭露 M1 的最高 High / 最低 Low。\n"
            "Range % = (High - Low) / replay_start 第一根 M1 Open × 100；不受 View Timeframe 影響。"
        )

        for w in (self.btn_h, self.btn_l, self.btn_t, self.btn_rect, self.btn_fibo):
            tb.addWidget(w)
        tb.addWidget(self.show_drawings_checkbox)
        tb.addWidget(self.show_orders_checkbox)
        for w in (self.btn_shot, self.btn_auto, self.btn_auto_all, self.timeframe_combo):
            tb.addWidget(w)
        tb.addWidget(QtWidgets.QLabel("X刻度"))
        tb.addWidget(self.x_tick_combo)
        tb.addWidget(self.timezone_combo)
        tb.addSpacing(8)
        tb.addWidget(self.vol_n_label)
        tb.addWidget(self.vol_n_spin)
        tb.addWidget(self.vol_stats_label)
        tb.addWidget(self.current_range_label)
        self.measure_status = QtWidgets.QLabel("MEASURE")
        self.measure_status.setStyleSheet(
            "background-color:#2a3952; border:1px solid #ffcc80; border-radius:3px; "
            "color:#ffcc80; padding:3px 8px; font-weight:700;"
        )
        self.measure_status.setVisible(False)
        tb.addWidget(self.measure_status)
        tb.addStretch(1)
        outer.addWidget(self.toolbar)

        self.info_panel = QtWidgets.QWidget()
        self.info_panel.setStyleSheet("background-color:#121722; border:1px solid #2a3142; border-radius:4px;")
        il = QtWidgets.QVBoxLayout(self.info_panel)
        il.setContentsMargins(10, 5, 10, 5)
        self.symbol_label = QtWidgets.QLabel("No Case")
        self.symbol_label.setStyleSheet("font-size:12pt; font-weight:700; color:#f5f7fa;")
        self.ohlc_label = QtWidgets.QLabel("開=--  高=--  低=--  收=--  漲跌=--")
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

        # Temporary measure overlay. It is visible only while the right button is held
        # inside Measure Mode and is never written to drawings[].
        self.measure_line = pg.PlotDataItem(pen=pg.mkPen((245, 245, 245), width=1))
        self.measure_label = pg.TextItem(
            text="", color=(245, 245, 245), anchor=(0, 1),
            fill=pg.mkBrush(13, 20, 32, 220), border=pg.mkPen(107, 127, 161, 180),
        )
        self.measure_line.setZValue(50)
        self.measure_label.setZValue(51)
        self.plot.addItem(self.measure_line, ignoreBounds=True)
        self.plot.addItem(self.measure_label, ignoreBounds=True)
        self.measure_line.hide()
        self.measure_label.hide()

        # Selected intraday-note callout.  This is a transient chart overlay, not
        # a Drawing Domain object and never persists to Case JSON.
        self.note_callout_line = pg.PlotDataItem(
            pen=pg.mkPen((255, 220, 40, 235), width=1.5)
        )
        self.note_callout_label = pg.TextItem(
            text="", color=(245, 247, 250), anchor=(0, 0),
            fill=pg.mkBrush(15, 20, 30, 235), border=pg.mkPen(255, 220, 40, 210),
        )
        self.note_callout_line.setZValue(70)
        self.note_callout_label.setZValue(71)
        self.plot.addItem(self.note_callout_line, ignoreBounds=True)
        self.plot.addItem(self.note_callout_label, ignoreBounds=True)
        self.note_callout_line.hide()
        self.note_callout_label.hide()

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
        self.show_drawings_checkbox.toggled.connect(self.set_all_drawings_visible)
        self.show_orders_checkbox.toggled.connect(self.set_all_orders_visible)
        self.btn_auto.clicked.connect(self.auto_scale)
        self.btn_auto_all.clicked.connect(self.auto_all)
        self.btn_shot.clicked.connect(self.export_screenshot)
        self.timeframe_combo.currentTextChanged.connect(self._tf_changed)
        self.x_tick_combo.currentTextChanged.connect(self._x_tick_changed)
        self.timezone_combo.currentTextChanged.connect(self._tz_changed)
        self.vol_n_spin.valueChanged.connect(self._update_volatility_stats)
        self.plot.scene().sigMouseClicked.connect(self._scene_clicked)
        self.plot.sigRangeChanged.connect(self._on_view_range_changed)
        self._mouse_proxy = pg.SignalProxy(self.graphics.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)
        self.graphics.setFocusPolicy(QtCore.Qt.StrongFocus)
        try:
            self.graphics.viewport().setMouseTracking(True)
            self.graphics.viewport().installEventFilter(self)
        except Exception:
            pass

        # Drawing clipboard shortcuts are chart-local so Ctrl+C/V in Notes, Pattern,
        # Order fields, etc. keep their normal text-editing behavior.
        self.shortcut_copy_drawing = QtGui.QShortcut(QtGui.QKeySequence.Copy, self)
        self.shortcut_copy_drawing.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcut_copy_drawing.activated.connect(self.copy_selected_drawing)
        self.shortcut_paste_drawing = QtGui.QShortcut(QtGui.QKeySequence.Paste, self)
        self.shortcut_paste_drawing.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcut_paste_drawing.activated.connect(self.paste_copied_drawing)

    def set_context(self, case, raw_df: pd.DataFrame, replay, case_path: str):
        self.case = case
        self.raw_df = raw_df
        self.replay = replay
        self.current_case_path = case_path
        self.selected_drawing_id = None
        self.selected_intraday_note = None
        self.show_all_intraday_notes = False
        self.show_all_drawings = True
        self.show_drawings_checkbox.blockSignals(True)
        self.show_drawings_checkbox.setChecked(True)
        self.show_drawings_checkbox.blockSignals(False)
        self._clear_note_callout_overlay()
        self._clear_all_note_callout_overlays()
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
        self.show_previous_rth = bool(case.display.get("show_previous_rth", False))
        self._update_volatility_stats()
        self._update_current_replay_range()

        self.rebuild_drawings()
        self.render(reset_x=True)

    def set_tool(self, name: str | None):
        if not self.drawing_interaction_enabled:
            self.tool_mode = None
            self.pending_point = None
            return
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

    def _update_volatility_stats(self, *_args):
        """Refresh N-day range summary from Enricher data already stored in Case JSON."""
        if self.case is None:
            self.vol_stats_label.setText("Med -- | σ -- | 2σ -- | 3σ --")
            self.vol_stats_label.setToolTip("尚未載入 Case")
            return
        payload = getattr(self.case, "reference_statistics", {}).get("intraday_volatility_20d")
        summary = summarize_intraday_volatility_payload(payload, int(self.vol_n_spin.value()))
        if summary is None:
            self.vol_stats_label.setText("Med -- | σ -- | 2σ -- | 3σ --")
            self.vol_stats_label.setToolTip("此 Case 沒有足夠的 Enricher 日內波動資料；請先執行 Case Enricher")
            return
        self.vol_stats_label.setText(
            f"Med {summary['median_range_pct']:.3f}% | "
            f"σ {summary['std_1x_range_pct']:.3f}% | "
            f"2σ {summary['std_2x_range_pct']:.3f}% | "
            f"3σ {summary['std_3x_range_pct']:.3f}%"
        )
        self.vol_stats_label.setToolTip(
            f"最近 {summary['session_count']} 個完整 RTH Session："
            f"{summary['start_date']} → {summary['end_date']}\n"
            "Range % = (High-Low)/RTH Open × 100；σ 為樣本標準差 ddof=1"
        )

    def _update_current_replay_range(self):
        """Refresh replay_start → current M1 range; transient and never persisted."""
        if self.replay is None or self.raw_df is None or self.raw_df.empty:
            self.current_range_label.setText("目前 Range --")
            return
        summary = summarize_current_replay_range(self.raw_df, float(self.replay.replay_start_ts), float(self.replay.current_ts))
        if summary is None:
            self.current_range_label.setText("目前 Range --")
            return
        self.current_range_label.setText(f"目前 Range {summary['range_pct']:.3f}% ({summary['range_points']:.2f})")
        self.current_range_label.setToolTip(
            "Replay 即時波幅（M1 revealed data）\n"
            f"Open {summary['base_open']:.2f} | High {summary['high']:.2f} | Low {summary['low']:.2f}\n"
            f"Range {summary['range_points']:.2f} = {summary['range_pct']:.3f}% | bars {summary['bar_count']}"
        )

    def set_order_events(self, events: list[dict], segments: list[dict] | None = None, render: bool = True):
        self.order_events = list(events or [])
        self.order_segments = list(segments or [])
        if render and self.replay is not None:
            self.render(reset_x=False)

    def set_pending_bracket(self, bracket: dict | None, render: bool = True):
        """Set the transient Entry + SL/TP plan shown before an order is filled."""
        self.pending_bracket = dict(bracket) if isinstance(bracket, dict) else None
        if render and self.replay is not None:
            self.render(reset_x=False)
            self._ensure_pending_bracket_visible()

    def _ensure_pending_bracket_visible(self):
        """Expand only the Y range when a newly created bracket sits off-screen."""
        try:
            group = self.pending_bracket["groups"][0]
            prices = [
                float(self.pending_bracket["entry_price"]),
                float(group["stop_price"]),
                float(group["target_price"]),
            ]
            current_low, current_high = sorted(self.plot.viewRange()[1])
        except (KeyError, TypeError, ValueError, IndexError):
            return
        if current_low <= min(prices) and max(prices) <= current_high:
            return
        span = max(max(prices) - min(prices), current_high - current_low, 1e-9)
        padding = span * 0.08
        self.plot.setYRange(min(current_low, min(prices)) - padding, max(current_high, max(prices)) + padding, padding=0)

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
        self._clear_pending_qty_controls()
        self.plot.clear()
        # Dynamic all-note callouts were removed by plot.clear(); drop stale refs.
        self._all_note_callout_items = []
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot.addItem(self.hline, ignoreBounds=True)
        self.plot.addItem(self.measure_line, ignoreBounds=True)
        self.plot.addItem(self.measure_label, ignoreBounds=True)
        self.plot.addItem(self.note_callout_line, ignoreBounds=True)
        self.plot.addItem(self.note_callout_label, ignoreBounds=True)
        self.vline.hide(); self.hline.hide()
        if not self._measure_dragging:
            self.measure_line.hide(); self.measure_label.hide()
        self.price_coord_label.hide(); self.time_coord_label.hide()
        self.btn_short_axis.hide(); self.btn_long_axis.hide()

        bars = self.visible_bars()
        self._last_bars = bars
        if not bars.empty:
            self._draw_bars(bars)
            self._set_ohlc_row(bars.iloc[-1])
        else:
            self._set_ohlc_row(None)

        self._render_reference_levels()
        self._render_drawing_items()
        self._render_order_overlay()
        self._render_pending_bracket()

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

        self._position_pending_qty_controls()
        self._update_current_replay_range()
        self._update_note_callout_overlay()


    def set_all_drawings_visible(self, enabled: bool):
        """Transient visibility toggle for all Drawing view objects."""
        self.show_all_drawings = bool(enabled)
        if not self.show_all_drawings:
            self.selected_drawing_id = None
        if self.replay is not None:
            self.render(reset_x=False)

    def set_all_orders_visible(self, enabled: bool):
        """Toggle transient order markers and completed trade segments."""
        self.show_all_orders = bool(enabled)
        if self.replay is not None:
            self.render(reset_x=False)

    def set_all_intraday_notes_visible(self, enabled: bool):
        """Show every visible/revealed note callout, or return to no bulk overlay."""
        self.show_all_intraday_notes = bool(enabled)
        self._update_note_callout_overlay()

    def set_selected_intraday_note(self, note):
        self.selected_intraday_note = dict(note) if isinstance(note, dict) else None
        self._update_note_callout_overlay()

    def _clear_note_callout_overlay(self):
        try:
            self.note_callout_line.setData([], [])
            self.note_callout_line.hide()
            self.note_callout_label.setText("")
            self.note_callout_label.hide()
        except Exception:
            pass

    def _clear_all_note_callout_overlays(self):
        for item in list(getattr(self, "_all_note_callout_items", [])):
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self._all_note_callout_items = []

    def _resolve_note_anchor(self, note):
        """Return displayed-candle X + exact M1-close Y for a revealed note."""
        if self.case is None or self.replay is None or not isinstance(note, dict):
            return None
        fallback_tz = self.case.time_context.get("case_timezone", self.case.display.get("timezone", "UTC"))
        note_ts = resolve_note_timestamp(note, fallback_timezone=fallback_tz)
        if note_ts is None or float(note_ts) > float(self.replay.current_ts) + 0.5:
            return None
        anchor = find_m1_close(self.raw_df, float(note_ts))
        if anchor is None:
            return None
        m1_x, anchor_y = anchor
        try:
            tf_sec = float(timeframe_seconds(self.timeframe_combo.currentText()))
            anchor_x = align_note_x_to_timeframe(float(m1_x), tf_sec)
        except Exception:
            anchor_x = float(m1_x)
        return float(anchor_x), float(anchor_y)

    def _render_all_note_callouts(self):
        self._clear_note_callout_overlay()
        self._clear_all_note_callout_overlays()
        if self.case is None or self.replay is None:
            return
        try:
            x_range, y_range = self.plot.viewRange()
            x_min, x_max = float(x_range[0]), float(x_range[1])
            y_min, y_max = float(y_range[0]), float(y_range[1])
        except Exception:
            return
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, max(abs(y_max), abs(y_min), 1.0) * 0.001)

        try:
            scene_rect = self.plot.getViewBox().sceneBoundingRect()
            view_width_px = max(float(scene_rect.width()), 480.0)
        except Exception:
            view_width_px = 900.0

        visible = []
        layout_inputs = []
        for note in list(getattr(self.case, "intraday_notes", []) or []):
            anchor = self._resolve_note_anchor(note)
            if anchor is None:
                continue
            anchor_x, anchor_y = anchor
            if anchor_x < x_min or anchor_x > x_max:
                continue
            # Bulk mode uses a slightly more compact wrap than the selected-note
            # callout so several annotations can coexist without dominating the chart.
            text = wrap_note_text(note.get("text", ""), width=20)
            if not text:
                continue
            lines = text.splitlines() or [text]
            longest = max((len(line) for line in lines), default=1)
            width_px = min(max(92.0 + longest * 4.6, 118.0), 245.0)
            width_norm = min(max(width_px / view_width_px, 0.10), 0.34)
            x_norm = min(max((anchor_x - x_min) / x_span, 0.0), 1.0)
            y_norm = min(max((anchor_y - y_min) / y_span, 0.0), 1.0)
            visible.append((anchor_x, anchor_y, text))
            layout_inputs.append({
                "x_norm": x_norm,
                "y_norm": y_norm,
                "width_norm": width_norm,
            })

        if not visible:
            return

        placements = layout_alternating_note_callouts(layout_inputs, edge_margin=0.07)
        # v1.20: deterministic Top / Bottom / Top / Bottom layout.  Labels stay
        # in the outer safe zones and a visible anchor dot marks the exact M1-close
        # point used by the note.  This intentionally favors readability over the
        # shortest possible leader line.
        edge_margin_y = 0.020

        for (anchor_x, anchor_y, text), placement in zip(visible, placements):
            side = placement["side"]
            label_x = x_min + float(placement["center_x_norm"]) * x_span

            if side == "top":
                label_y = y_max - y_span * edge_margin_y
                label_anchor = (0.5, 0.0)
                line_end_y = label_y - y_span * 0.012
            else:
                label_y = y_min + y_span * edge_margin_y
                label_anchor = (0.5, 1.0)
                line_end_y = label_y + y_span * 0.012

            line = pg.PlotDataItem(
                x=[anchor_x, label_x], y=[anchor_y, line_end_y],
                pen=pg.mkPen((255, 220, 40, 235), width=1.25),
            )
            point = pg.ScatterPlotItem(
                x=[anchor_x], y=[anchor_y], size=8,
                symbol="o",
                pen=pg.mkPen((255, 255, 255, 245), width=1.0),
                brush=pg.mkBrush(255, 220, 40, 245),
            )
            label = pg.TextItem(
                text=text, color=(245, 247, 250), anchor=label_anchor,
                fill=pg.mkBrush(15, 20, 30, 238), border=pg.mkPen(255, 220, 40, 215),
            )
            line.setZValue(70)
            point.setZValue(72)
            label.setZValue(71)
            label.setPos(label_x, label_y)
            self.plot.addItem(line, ignoreBounds=True)
            self.plot.addItem(point, ignoreBounds=True)
            self.plot.addItem(label, ignoreBounds=True)
            self._all_note_callout_items.extend([line, point, label])

    def _update_note_callout_overlay(self):
        if self.show_all_intraday_notes:
            self._render_all_note_callouts()
            return

        self._clear_all_note_callout_overlays()
        if self.case is None or self.replay is None or self.selected_intraday_note is None:
            self._clear_note_callout_overlay()
            return

        anchor = self._resolve_note_anchor(self.selected_intraday_note)
        if anchor is None:
            self._clear_note_callout_overlay()
            return
        anchor_x, anchor_y = anchor

        try:
            x_range, y_range = self.plot.viewRange()
            x_min, x_max = float(x_range[0]), float(x_range[1])
            y_min, y_max = float(y_range[0]), float(y_range[1])
        except Exception:
            self._clear_note_callout_overlay()
            return

        if anchor_x < x_min or anchor_x > x_max:
            self._clear_note_callout_overlay()
            return

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, max(abs(y_max), abs(y_min), 1.0) * 0.001)
        label_x = x_min + x_span * 0.035
        label_y = y_max - y_span * 0.035
        line_end_x = label_x + x_span * 0.10
        line_end_y = label_y - y_span * 0.035

        text = wrap_note_text(self.selected_intraday_note.get("text", ""), width=26)
        if not text:
            self._clear_note_callout_overlay()
            return

        self.note_callout_line.setData([anchor_x, line_end_x], [anchor_y, line_end_y])
        self.note_callout_line.show()
        self.note_callout_label.setText(text)
        self.note_callout_label.setPos(label_x, label_y)
        self.note_callout_label.show()

    def set_previous_rth_visible(self, enabled: bool):
        self.show_previous_rth = bool(enabled)
        if self.replay is not None:
            self.render(reset_x=False)

    def _render_reference_levels(self):
        if not self.show_previous_rth or self.case is None:
            return
        rth = getattr(self.case, "reference_levels", {}).get("previous_rth")
        if not isinstance(rth, dict):
            return
        try:
            high = float(rth["high"])
            low = float(rth["low"])
            close = float(rth["close"])
        except Exception:
            return

        high_pen = pg.mkPen((255, 221, 87, 215), width=1, style=QtCore.Qt.DashLine)
        low_pen = pg.mkPen((255, 221, 87, 215), width=1, style=QtCore.Qt.DashLine)
        close_pen = pg.mkPen((255, 255, 255, 200), width=1, style=QtCore.Qt.DashLine)

        entries = [
            ("H", high, high_pen, f"Previous RTH High: {high:.2f}", "#ffdd57"),
            ("L", low, low_pen, f"Previous RTH Low: {low:.2f}", "#ffdd57"),
            ("C", close, close_pen, f"Previous RTH Close: {close:.2f}", "#ffffff"),
        ]

        def _fmt(v: float) -> str:
            s = f"{float(v):.2f}"
            return s.rstrip("0").rstrip(".")

        try:
            x_range = self.plot.viewRange()[0]
            right_x = float(x_range[1])
            left_x = float(x_range[0])
        except Exception:
            right_x = float(self.replay.current_ts)
            left_x = float(self.replay.data_start_ts)
        x_pad = max(60.0, (right_x - left_x) * 0.01)
        label_x = right_x - x_pad

        for prefix, y, pen, tooltip, color in entries:
            line = pg.InfiniteLine(pos=y, angle=0, movable=False, pen=pen)
            line.setZValue(2)
            line.setToolTip(tooltip)
            self.plot.addItem(line)

            label = pg.TextItem(text=f"{prefix} {_fmt(y)}", color=color, anchor=(1, 1))
            label.setPos(label_x, y)
            label.setZValue(3)
            self.plot.addItem(label)

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

    def _render_order_overlay(self):
        if not self.show_all_orders:
            return
        tf_sec = float(timeframe_seconds(self.timeframe_combo.currentText()))
        def display_x(ts: float):
            return math.floor(ts / tf_sec) * tf_sec if tf_sec > 60 else ts
        for segment in self.order_segments:
            pnl = float(segment.get("pnl") or 0.0)
            if abs(pnl) <= 1e-12:
                continue
            color = (255, 230, 0) if pnl > 0 else (255, 64, 64)
            item = pg.PlotDataItem(
                x=[display_x(float(segment["entry_timestamp"])), display_x(float(segment["exit_timestamp"]))],
                y=[float(segment["entry_price"]), float(segment["exit_price"])],
                pen=pg.mkPen(color=color, width=1, style=QtCore.Qt.DotLine),
            )
            item.setZValue(20)
            self.plot.addItem(item)

        longs = [e for e in self.order_events if str(e.get("side")) == "long"]
        shorts = [e for e in self.order_events if str(e.get("side")) == "short"]
        def marker_x(event):
            return display_x(float(event["timestamp"]))
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

    def _render_pending_bracket(self):
        """Render the session-only, draggable Entry / SL / TP bracket layout."""
        bracket = self.pending_bracket
        if not bracket:
            return
        try:
            side = str(bracket["side"])
            entry = float(bracket["entry_price"])
            qty = float(bracket["qty"])
            groups = list(bracket["groups"])
        except (KeyError, TypeError, ValueError):
            return
        if not groups:
            return

        entry_color = (255, 82, 95) if side == "short" else (31, 121, 245)
        entry_item = pg.InfiniteLine(
            pos=entry,
            angle=0,
            movable=True,
            pen=pg.mkPen(color=entry_color, width=1, style=QtCore.Qt.DashLine),
        )
        entry_item.sigPositionChangeFinished.connect(
            lambda item=entry_item: self._pending_entry_drag_finished(item)
        )
        entry_item.setZValue(30)
        self.plot.addItem(entry_item)
        self._add_pending_qty_control(
            f"Entry {qty:.4f}", entry, entry_color,
            lambda: self._edit_pending_entry_qty(),
        )

        for index, group in enumerate(groups, start=1):
            try:
                group_id = str(group["id"])
                group_qty = float(group["qty"])
                stop = float(group["stop_price"])
                target = float(group["target_price"])
            except (KeyError, TypeError, ValueError):
                continue
            for leg, price, color in (
                ("stop", stop, (255, 179, 0)),
                ("target", target, (0, 196, 168)),
            ):
                item = pg.InfiniteLine(
                    pos=price,
                    angle=0,
                    movable=True,
                    pen=pg.mkPen(color=color, width=1, style=QtCore.Qt.DashLine),
                )
                item.sigPositionChangeFinished.connect(
                    lambda line=item, group_id=group_id, leg=leg: self._pending_leg_drag_finished(line, group_id, leg)
                )
                item.setZValue(30)
                self.plot.addItem(item)
            self._add_pending_qty_control(
                f"SL {group_qty:.4f}", stop, (255, 179, 0),
                lambda gid=group_id: self._edit_pending_group_qty(gid),
            )
            self._add_pending_qty_control(
                f"TP {group_qty:.4f}", target, (0, 196, 168),
                lambda gid=group_id: self._edit_pending_group_qty(gid),
            )

    def _clear_pending_qty_controls(self):
        for button, _price in self._pending_qty_controls:
            button.hide()
            button.deleteLater()
        self._pending_qty_controls = []

    def _add_pending_qty_control(self, text: str, price: float, color, callback=None):
        """Create the boxed lot label attached to a pending-order line."""
        control = QtWidgets.QPushButton(text, self.graphics) if callback else QtWidgets.QLabel(text, self.graphics)
        control.setFixedHeight(22)
        control.setStyleSheet(
            f"background-color:#0d1420; border:1px solid rgb{tuple(color)}; border-radius:2px; "
            f"color:rgb{tuple(color)}; padding:1px 5px; font-weight:600;"
        )
        if callback:
            control.setToolTip("點擊修改 lots；同組 SL 與 TP 會同步")
            control.clicked.connect(callback)
        else:
            control.setToolTip("SL/TP lots 由 Entry 分配；不可直接修改")
            control.setAlignment(QtCore.Qt.AlignCenter)
        control.hide()
        self._pending_qty_controls.append((control, float(price)))

    def _position_pending_qty_controls(self):
        if not self.pending_bracket:
            return
        try:
            x_range = self.plot.vb.viewRange()[0]
            reference_x = (float(x_range[0]) + float(x_range[1])) * 0.5
        except Exception:
            return
        for button, price in self._pending_qty_controls:
            try:
                scene = self.plot.vb.mapViewToScene(QtCore.QPointF(reference_x, price))
                widget_y = self.graphics.mapFromScene(scene).y()
            except Exception:
                button.hide()
                continue
            x = max(0, self.graphics.width() - 72 - button.sizeHint().width() - 22)
            y = int(widget_y - button.height() * 0.5)
            if 0 <= y <= self.graphics.height() - button.height():
                button.move(x, y)
                button.show()
                button.raise_()
            else:
                button.hide()

    def _edit_pending_entry_qty(self):
        if not self.pending_bracket:
            return
        try:
            allocated = float(self.pending_bracket["qty"]) - available_bracket_qty(self.pending_bracket)
            current = float(self.pending_bracket["qty"])
        except (KeyError, TypeError, ValueError):
            return
        qty, accepted = QtWidgets.QInputDialog.getDouble(
            self, "Entry lots", "Lots", current, max(allocated, 0.0001), 1_000_000.0, 4,
        )
        if accepted:
            self.pending_bracket_edited.emit(set_bracket_qty(self.pending_bracket, qty))

    def _edit_pending_group_qty(self, group_id: str):
        if not self.pending_bracket:
            return
        group = next((g for g in self.pending_bracket.get("groups", []) if str(g.get("id")) == group_id), None)
        if group is None:
            return
        try:
            current = float(group["qty"])
            maximum = current + available_bracket_qty(self.pending_bracket)
        except (KeyError, TypeError, ValueError):
            return
        qty, accepted = QtWidgets.QInputDialog.getDouble(
            self, "SL/TP lots", "Lots", current, 0.0001, maximum, 4,
        )
        if accepted:
            self.pending_bracket_edited.emit(set_bracket_group_qty(self.pending_bracket, group_id, qty))

    def _pending_entry_drag_finished(self, item):
        if not self.pending_bracket:
            return
        try:
            self.pending_bracket_edited.emit(move_bracket_entry(self.pending_bracket, float(item.value())))
        except (TypeError, ValueError, KeyError):
            return

    def _pending_leg_drag_finished(self, item, group_id: str, leg: str):
        if not self.pending_bracket:
            return
        try:
            self.pending_bracket_edited.emit(move_bracket_leg(self.pending_bracket, group_id, leg, float(item.value())))
        except (TypeError, ValueError, KeyError):
            return

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
        self._position_pending_qty_controls()

        # Keep Fibonacci 0/1 anchor hit-zones roughly constant in screen pixels.
        # This keeps the two selected anchor points easy to grab at any zoom level.
        for did, group in list(self.drawing_items.items()):
            if isinstance(group, dict) and group.get("type") == "fibonacci":
                self._update_fibo_view_geometry(did, update_box=False)
            elif isinstance(group, dict) and group.get("type") == "text":
                # Text wrapping is based on the box width in screen pixels, so refresh it after zoom/pan.
                self._update_text_box_view(did, update_roi=False)

        self._update_note_callout_overlay()

    def export_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "匯出圖表", "chart.png", "PNG (*.png)")
        if not path:
            return
        self.grab().save(path)

    @staticmethod
    def _qt_mouse_pos(event):
        """Return a QPoint compatible with Qt5/Qt6 mouse events."""
        try:
            return event.position().toPoint()
        except Exception:
            try:
                return event.pos()
            except Exception:
                return QtCore.QPoint()

    def _viewport_event_to_scene(self, event):
        try:
            return self.graphics.mapToScene(self._qt_mouse_pos(event))
        except Exception:
            return None

    def _set_measure_mode(self, enabled: bool):
        self.measure_mode = bool(enabled)
        self.measure_status.setVisible(self.measure_mode)
        self._clear_measure_overlay()
        try:
            self.graphics.viewport().setCursor(
                QtCore.Qt.CrossCursor if self.measure_mode else QtCore.Qt.ArrowCursor
            )
        except Exception:
            pass

    def _toggle_measure_mode(self):
        self._set_measure_mode(not self.measure_mode)

    def _clear_measure_overlay(self):
        self._measure_dragging = False
        self._measure_start = None
        self._measure_end = None
        try:
            self.graphics.viewport().releaseMouse()
        except Exception:
            pass
        try:
            self.measure_line.setData([], [])
            self.measure_line.hide()
            self.measure_label.setText("")
            self.measure_label.hide()
        except Exception:
            pass

    @staticmethod
    def _measure_text(start_price: float, end_price: float) -> str:
        if abs(float(end_price)) < 10:
            price_text = f"{float(end_price):.4f}"
        else:
            price_text = f"{float(end_price):.2f}"
        if abs(float(start_price)) < 1e-12:
            pct = 0.0
        else:
            pct = (float(end_price) - float(start_price)) / float(start_price) * 100.0
        return f"{price_text}   {pct:+.2f}%"

    def _measure_begin(self, scene_pos) -> bool:
        if scene_pos is None or not self.plot.sceneBoundingRect().contains(scene_pos):
            return False
        point = self.plot.vb.mapSceneToView(scene_pos)
        self._measure_start = (float(point.x()), float(point.y()))
        self._measure_end = self._measure_start
        self._measure_dragging = True
        try:
            self.graphics.viewport().grabMouse()
        except Exception:
            pass
        self._update_measure_overlay(self._measure_start[0], self._measure_start[1])
        return True

    def _update_measure_overlay(self, x: float, y: float):
        if not self._measure_dragging or self._measure_start is None:
            return
        sx, sy = self._measure_start
        ex, ey = float(x), float(y)
        self._measure_end = (ex, ey)
        self.measure_line.setData([sx, ex], [sy, ey])
        self.measure_line.show()
        self.measure_label.setText(self._measure_text(sy, ey))
        self.measure_label.setPos(ex, ey)
        self.measure_label.show()

    def eventFilter(self, watched, event):
        """Middle click toggles Measure Mode; right drag performs transient measurement."""
        try:
            viewport = self.graphics.viewport()
        except Exception:
            viewport = None
        if watched is not viewport:
            return super().eventFilter(watched, event)

        et = event.type()
        mouse_press = QtCore.QEvent.MouseButtonPress
        mouse_release = QtCore.QEvent.MouseButtonRelease
        mouse_move = QtCore.QEvent.MouseMove

        if et == mouse_press and event.button() == QtCore.Qt.MiddleButton:
            self._toggle_measure_mode()
            event.accept()
            return True

        if not self.measure_mode:
            return super().eventFilter(watched, event)

        if et == mouse_press and event.button() == QtCore.Qt.RightButton:
            scene_pos = self._viewport_event_to_scene(event)
            self._measure_begin(scene_pos)
            event.accept()
            return True

        if et == mouse_move and self._measure_dragging:
            # Right-button ownership remains with Measure Mode until release.
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
            # The requested behavior is deliberately ephemeral: release means disappear.
            self._clear_measure_overlay()
            event.accept()
            return True

        return super().eventFilter(watched, event)

    def _set_ohlc_row(self, row):
        """Show OHLC for one rendered candle. None resets the information row."""
        self.ohlc_label.setText(format_ohlc_text(row))

    def _restore_latest_ohlc(self):
        if self._last_bars.empty:
            self._set_ohlc_row(None)
        else:
            self._set_ohlc_row(self._last_bars.iloc[-1])

    def _mouse_moved(self, evt):
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            self.vline.hide(); self.hline.hide()
            self.price_coord_label.hide(); self.time_coord_label.hide()
            self.btn_short_axis.hide(); self.btn_long_axis.hide()
            self._restore_latest_ohlc()
            return
        p = self.plot.vb.mapSceneToView(pos)
        x = float(p.x())
        y = float(p.y())
        hovered_row = None

        # TradingView-like crosshair: X snaps to nearest revealed candle.
        # The OHLC information row follows that exact displayed candle instead of the latest candle.
        if not self._last_bars.empty:
            ts = self._last_bars["timestamp"].to_numpy(dtype=float)
            idx = int(np.argmin(np.abs(ts - x)))
            x = float(ts[idx])
            hovered_row = self._last_bars.iloc[idx]
        if self._ctrl_pressed():
            x, y = self._magnet_snap_point(x, y)
            if not self._last_bars.empty:
                ts = self._last_bars["timestamp"].to_numpy(dtype=float)
                idx = int(np.argmin(np.abs(ts - x)))
                hovered_row = self._last_bars.iloc[idx]
        self._set_ohlc_row(hovered_row)
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

    def _drawing_id_near_scene_pos(self, scene_pos, tolerance_px: float = 8.0) -> str | None:
        """Pick the nearest Drawing by its *visible geometry* in screen pixels.

        This is intentionally independent from QGraphicsItem hit-boxes / z-order.
        In particular, Fibonacci is selectable only near one of its visible level
        segments, never merely because the pointer is inside the Fibo's bounding
        region. Rectangle selection uses its four edges; Text Box uses its body.
        """
        if self.case is None:
            return None
        try:
            px = float(scene_pos.x())
            py = float(scene_pos.y())
        except Exception:
            return None

        def scene_xy(x: float, y: float) -> tuple[float, float]:
            point = self.plot.vb.mapViewToScene(QtCore.QPointF(float(x), float(y)))
            return float(point.x()), float(point.y())

        candidates: list[tuple[float, int, str]] = []
        drawings = list(getattr(self.case, "drawings", []) or [])
        try:
            view_x = self.plot.vb.mapSceneToView(scene_pos)
            pointer_view_x = float(view_x.x())
        except Exception:
            pointer_view_x = 0.0

        for order, drawing in enumerate(drawings):
            did = drawing.get("id")
            dtype = drawing.get("type")
            if not did:
                continue
            distance = float("inf")
            try:
                if dtype == "horizontal_line":
                    # Infinite horizontal line: distance is purely vertical in screen pixels.
                    _sx, sy = scene_xy(pointer_view_x, float(drawing["price"]))
                    distance = abs(py - sy)

                elif dtype == "trend_line":
                    p1, p2 = drawing["points"]
                    ax, ay = scene_xy(float(p1["time"]), float(p1["price"]))
                    bx, by = scene_xy(float(p2["time"]), float(p2["price"]))
                    distance = point_to_segment_distance(px, py, ax, ay, bx, by)

                elif dtype == "rectangle":
                    left, right, bottom, top = self._rectangle_geometry(drawing)
                    a = scene_xy(left, top)
                    b = scene_xy(right, bottom)
                    distance = point_to_rect_distance(
                        px, py, a[0], a[1], b[0], b[1], interior_is_hit=False
                    )

                elif dtype == "text":
                    left, right, bottom, top = self._text_box_geometry(drawing)
                    a = scene_xy(left, top)
                    b = scene_xy(right, bottom)
                    distance = point_to_rect_distance(
                        px, py, a[0], a[1], b[0], b[1], interior_is_hit=True
                    )

                elif dtype == "fibonacci":
                    sx, _sy, ex, _ey, _left, _right, _ymin, _ymax, ys = self._fibo_geometry(drawing)
                    # Crucial: only the rendered horizontal level segments count.
                    # No Fibo bounding-box / ROI interior participates in selection.
                    distances = []
                    for y in ys:
                        ax, ay = scene_xy(sx, y)
                        bx, by = scene_xy(ex, y)
                        distances.append(point_to_segment_distance(px, py, ax, ay, bx, by))
                    if distances:
                        distance = min(distances)
            except Exception:
                continue

            if distance <= float(tolerance_px):
                # Later drawings win only as a tie-breaker; geometric distance is primary.
                candidates.append((float(distance), -int(order), str(did)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

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

    def _shift_axis_locked_point(self, anchor: tuple[float, float], moving: tuple[float, float]) -> tuple[float, float]:
        """Lock a moving line endpoint horizontally/vertically relative to anchor using screen-space direction."""
        ax, ay = float(anchor[0]), float(anchor[1])
        mx, my = float(moving[0]), float(moving[1])
        try:
            a_scene = self.plot.vb.mapViewToScene(QtCore.QPointF(ax, ay))
            m_scene = self.plot.vb.mapViewToScene(QtCore.QPointF(mx, my))
            dx = abs(float(m_scene.x() - a_scene.x()))
            dy = abs(float(m_scene.y() - a_scene.y()))
        except Exception:
            # Screen mapping is preferred because X=time and Y=price use different units.
            dx = abs(mx - ax)
            dy = abs(my - ay)
        if dx >= dy:
            return mx, ay
        return ax, my

    @staticmethod
    def _shift_pressed() -> bool:
        try:
            return bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
        except Exception:
            return False

    @staticmethod
    def _ctrl_pressed() -> bool:
        try:
            return bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier)
        except Exception:
            return False

    def _magnet_snap_point(self, x: float, y: float) -> tuple[float, float]:
        """Snap to the nearest revealed candle and its nearest OHLC price. Never reads future bars."""
        if self._last_bars is None or self._last_bars.empty:
            return float(x), float(y)
        try:
            bars = self._last_bars
            ts = bars["timestamp"].to_numpy(dtype=float)
            idx = int(np.argmin(np.abs(ts - float(x))))
            row = bars.iloc[idx]
            snap_x = float(row["timestamp"])
            candidates = [float(row[k]) for k in ("open", "high", "low", "close") if k in row and pd.notna(row[k])]
            if not candidates:
                return snap_x, float(y)

            # Compare candidate prices in screen space so the choice remains natural at any zoom.
            try:
                cursor_scene = self.plot.vb.mapViewToScene(QtCore.QPointF(snap_x, float(y)))
                best = min(
                    candidates,
                    key=lambda price: abs(float(self.plot.vb.mapViewToScene(QtCore.QPointF(snap_x, price)).y() - cursor_scene.y())),
                )
            except Exception:
                best = min(candidates, key=lambda price: abs(price - float(y)))
            return snap_x, float(best)
        except Exception:
            return float(x), float(y)

    def _cursor_view_position(self) -> tuple[float, float] | None:
        try:
            widget_pos = self.graphics.mapFromGlobal(QtGui.QCursor.pos())
            scene_pos = self.graphics.mapToScene(widget_pos)
            point = self.plot.vb.mapSceneToView(scene_pos)
            return float(point.x()), float(point.y())
        except Exception:
            return None

    def _magnet_cursor_delta(self) -> tuple[float, float]:
        pos = self._cursor_view_position()
        if pos is None:
            return 0.0, 0.0
        x, y = pos
        sx, sy = self._magnet_snap_point(x, y)
        return float(sx - x), float(sy - y)

    @staticmethod
    def _line_state_points(item) -> list[tuple[float, float]]:
        """Return trend-line endpoints in absolute ViewBox coordinates.

        pyqtgraph LineSegmentROI stores handle points in ROI-local coordinates while
        whole-object dragging changes ROI.pos().  Persisting only getState()["points"]
        therefore makes a moved line jump back after the next render/save.  Always
        fold the ROI translation into the returned coordinates.
        """
        try:
            pts = item.getState().get("points", [])
            if len(pts) >= 2:
                pos = item.pos()
                ox, oy = float(pos.x()), float(pos.y())
                return [
                    (float(pts[0][0]) + ox, float(pts[0][1]) + oy),
                    (float(pts[1][0]) + ox, float(pts[1][1]) + oy),
                ]
        except Exception:
            pass
        return []

    def _set_line_roi_points(self, item, points: list[tuple[float, float]]):
        """Set absolute trend-line endpoints without losing the ROI translation."""
        try:
            pos = item.pos()
            ox, oy = float(pos.x()), float(pos.y())
        except Exception:
            ox, oy = 0.0, 0.0
        qpts = [QtCore.QPointF(float(x) - ox, float(y) - oy) for x, y in points]
        try:
            item.setPoints(qpts)
            return
        except Exception:
            pass
        try:
            handles = item.getHandles()
            for handle, point in zip(handles, qpts):
                handle.setPos(point)
        except Exception:
            pass

    def _enforce_trend_shift_constraint(self, item):
        """Apply live Ctrl magnet and Shift horizontal/vertical constraints to a trend line."""
        if getattr(item, "_shift_constraint_guard", False):
            return
        current = self._line_state_points(item)
        if len(current) != 2:
            return
        previous = getattr(item, "_shift_prev_points", None)
        if previous is None or len(previous) != 2:
            item._shift_prev_points = current
            return

        shift = self._shift_pressed()
        ctrl = self._ctrl_pressed()
        if not shift and not ctrl:
            item._shift_prev_points = current
            return

        def screen_dist(a, b):
            try:
                sa = self.plot.vb.mapViewToScene(QtCore.QPointF(float(a[0]), float(a[1])))
                sb = self.plot.vb.mapViewToScene(QtCore.QPointF(float(b[0]), float(b[1])))
                return ((float(sa.x()-sb.x()))**2 + (float(sa.y()-sb.y()))**2) ** 0.5
            except Exception:
                return ((float(a[0]-b[0]))**2 + (float(a[1]-b[1]))**2) ** 0.5

        d0 = screen_dist(current[0], previous[0])
        d1 = screen_dist(current[1], previous[1])
        whole_move = min(d0, d1) > 1.5 and max(d0, d1) < min(d0, d1) * 1.6
        adjusted = list(current)

        if whole_move:
            if ctrl:
                dx, dy = self._magnet_cursor_delta()
                adjusted = [(p[0] + dx, p[1] + dy) for p in current]
            # Shift does not alter a whole-line translation; it only constrains endpoint resizing.
        else:
            moved = 0 if d0 >= d1 else 1
            anchor = current[1 - moved]
            moving = current[moved]
            if ctrl:
                moving = self._magnet_snap_point(moving[0], moving[1])
            if shift:
                moving = self._shift_axis_locked_point(anchor, moving)
            adjusted[moved] = moving

        if adjusted == current:
            item._shift_prev_points = current
            return
        item._shift_constraint_guard = True
        try:
            self._set_line_roi_points(item, adjusted)
            item._shift_prev_points = adjusted
        finally:
            item._shift_constraint_guard = False

    def _scene_clicked(self, evt):
        if self.case is None:
            return
        if not self.drawing_interaction_enabled:
            return
        pos = evt.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return

        # Measure Mode owns right-click press/drag/release through the viewport event filter.
        # Never open a Drawing context menu while measuring.
        if self.measure_mode and evt.button() == QtCore.Qt.RightButton:
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
                did = self._drawing_id_near_scene_pos(pos, tolerance_px=8.0)
                self._select_drawing_id(did)
            return

        p = self.plot.vb.mapSceneToView(pos)
        x, y = float(p.x()), float(p.y())
        if self._ctrl_pressed():
            x, y = self._magnet_snap_point(x, y)

        if self.tool_mode == "horizontal_line":
            self.case.drawings.append({
                "id": f"drawing-{uuid.uuid4().hex[:12]}",
                "type": "horizontal_line",
                "price": y,
                "style": {"color": "#ffffff", "width": 2, "line_style": "solid"},
            })
            self.tool_mode = None
            self.case.touch(); self.dirty.emit(); self.history_committed.emit("Add Drawing"); self.rebuild_drawings(); self.render(False)
            return

        if self.tool_mode == "trend_line":
            if self.pending_point is None:
                self.pending_point = (x, y)
                return
            x1, y1 = self.pending_point
            if self._shift_pressed():
                x, y = self._shift_axis_locked_point((x1, y1), (x, y))
            self.case.drawings.append({
                "id": f"drawing-{uuid.uuid4().hex[:12]}",
                "type": "trend_line",
                "points": [{"time": x1, "price": y1}, {"time": x, "price": y}],
                "style": {"color": "#ffffff", "width": 2, "line_style": "solid"},
            })
            self.pending_point = None; self.tool_mode = None
            self.case.touch(); self.dirty.emit(); self.history_committed.emit("Add Drawing"); self.rebuild_drawings(); self.render(False)
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
            self.case.touch(); self.dirty.emit(); self.history_committed.emit("Add Drawing"); self.rebuild_drawings(); self.render(False)
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
            self.case.touch(); self.dirty.emit(); self.history_committed.emit("Add Drawing"); self.rebuild_drawings(); self.render(False)
            return

        if self.tool_mode == "text":
            defaults = self._default_text_style()
            dialog = TextSettingsDialog(
                text="",
                text_color=defaults["color"],
                font_size=int(defaults["font_size"]),
                bold=bool(defaults["bold"]),
                italic=bool(defaults["italic"]),
                border_enabled=bool(defaults["border_enabled"]),
                border_color=defaults["border_color"],
                border_width=int(defaults["border_width"]),
                background_enabled=bool(defaults["background_enabled"]),
                background_color=defaults["background_color"],
                background_opacity=int(defaults["background_opacity"]),
                auto_wrap=bool(defaults["auto_wrap"]),
                save_template_callback=lambda payload: self._save_template_interactive("text", payload),
                parent=self,
            )
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                text, style = dialog.values()
                if text.strip():
                    box_w, box_h = self._initial_text_box_size(text, style)
                    self.case.drawings.append({
                        "id": f"drawing-{uuid.uuid4().hex[:12]}",
                        "type": "text",
                        "time": x,
                        "price": y,
                        "text": text,
                        "box": {"width": box_w, "height": box_h},
                        "style": style,
                    })
                    self.case.touch(); self.dirty.emit(); self.history_committed.emit("Add Drawing"); self.rebuild_drawings(); self.render(False)
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

    @staticmethod
    def _default_text_style() -> dict:
        return {
            "color": "#ffffff",
            "font_size": 12,
            "bold": False,
            "italic": False,
            "border_enabled": False,
            "border_color": "#ffffff",
            "border_width": 1,
            "background_enabled": False,
            "background_color": "#181c27",
            "background_opacity": 0,
            "auto_wrap": True,
        }

    def _view_size_from_pixels(self, width_px: float, height_px: float) -> tuple[float, float]:
        """Convert a screen-pixel box size into current chart view coordinates."""
        try:
            x_per_px, y_per_px = self.plot.vb.viewPixelSize()
            width = max(abs(float(x_per_px)) * float(width_px), 1e-6)
            height = max(abs(float(y_per_px)) * float(height_px), 1e-9)
            if np.isfinite(width) and np.isfinite(height):
                return width, height
        except Exception:
            pass
        tf = float(timeframe_seconds(self.timeframe_combo.currentText()))
        return max(tf * 12.0, 1e-6), 1.0

    def _default_text_box_size(self) -> tuple[float, float]:
        # Use a predictable TradingView-like initial box rather than a percentage
        # of the current viewport.  This prevents newly-created text from starting
        # inside a tiny box when the chart is zoomed far out.
        return self._view_size_from_pixels(360.0, 180.0)

    @staticmethod
    def _text_font_from_style(style: dict) -> QtGui.QFont:
        font = QtGui.QFont()
        font.setPointSize(max(6, int(style.get("font_size", 12))))
        font.setBold(bool(style.get("bold", False)))
        font.setItalic(bool(style.get("italic", False)))
        return font

    @classmethod
    def _measure_no_wrap_text_pixels(cls, text: str, style: dict) -> tuple[float, float]:
        """Measure plain text without wrapping.

        Width follows the longest explicit line (split by newline), matching the
        requested TradingView-style no-wrap behaviour. A small inner padding is
        included so glyphs never touch the frame.
        """
        font = cls._text_font_from_style(style)
        metrics = QtGui.QFontMetricsF(font)
        lines = str(text or "").splitlines() or [""]
        longest = max((float(metrics.horizontalAdvance(line.expandtabs(4))) for line in lines), default=0.0)
        line_height = max(float(metrics.height()), 1.0)
        width_px = max(40.0, longest + 18.0)
        height_px = max(30.0, line_height * max(len(lines), 1) + 16.0)
        return width_px, height_px

    def _initial_text_box_size(self, text: str, style: dict) -> tuple[float, float]:
        """Return a predictable initial box while fully containing its text."""
        default_w, default_h = self._default_text_box_size()
        try:
            font = self._text_font_from_style(style)
            if not bool(style.get("auto_wrap", True)):
                required_w_px, required_h_px = self._measure_no_wrap_text_pixels(text, style)
                required_w, required_h = self._view_size_from_pixels(required_w_px, max(180.0, required_h_px))
                return max(required_w, 1e-6), max(default_h, required_h)

            # Wrapped text starts from the default 360 px width and grows in
            # height only when the initial content needs more room.
            doc = QtGui.QTextDocument()
            doc.setDefaultFont(font)
            doc.setPlainText(str(text or ""))
            doc.setTextWidth(344.0)  # 360 px box minus horizontal padding.
            required_h_px = max(180.0, float(doc.size().height()) + 18.0)
            _w, required_h = self._view_size_from_pixels(360.0, required_h_px)
            return default_w, max(default_h, required_h)
        except Exception:
            return default_w, default_h

    def _fit_text_box_width_to_content(self, drawing: dict):
        """When auto-wrap is disabled, fit box width to the longest explicit line."""
        style, box = self._normalize_text_drawing(drawing)
        if bool(style.get("auto_wrap", True)):
            return
        try:
            width_px, _height_px = self._measure_no_wrap_text_pixels(str(drawing.get("text", "")), style)
            width_view, _ = self._view_size_from_pixels(width_px, 32.0)
            box["width"] = max(float(width_view), 1e-6)
        except Exception:
            pass

    @staticmethod
    def _clear_roi_handles(roi):
        """Remove any existing ROI handles, including legacy corner handles."""
        try:
            for handle in list(roi.getHandles()):
                try:
                    roi.removeHandle(handle)
                except Exception:
                    try:
                        handle.setParentItem(None)
                        scene = handle.scene()
                        if scene is not None:
                            scene.removeItem(handle)
                    except Exception:
                        pass
        except Exception:
            pass

    def _make_standard_adjustment_marker(self, x: float, y: float, visible: bool = False):
        """Create the shared circular adjustment-point visual used by FIBO and ROI handles."""
        marker = pg.ScatterPlotItem(
            [float(x)], [float(y)],
            symbol="o",
            size=12,
            pen=pg.mkPen("#2d7cff", width=2),
            brush=pg.mkBrush(18, 24, 35, 230),
            pxMode=True,
        )
        try:
            marker.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        except Exception:
            pass
        marker.setZValue(31)
        marker.setVisible(bool(visible))
        self.plot.addItem(marker)
        return marker

    @staticmethod
    def _roi_handle_absolute_position(roi, handle):
        """Return an unrotated ROI handle position in plot/view coordinates."""
        try:
            rp = roi.pos()
            hp = handle.pos()
            return float(rp.x() + hp.x()), float(rp.y() + hp.y())
        except Exception:
            return None

    def _install_standard_roi_handle_markers(self, roi, selected: bool = False):
        """Overlay shared markers only for editable Analyzer Drawings.

        In Viewer mode native ROI handles are made inert/invisible and no custom
        adjustment markers are created at all.
        """
        old_markers = list(getattr(roi, "_standard_handle_markers", []) or [])
        for marker in old_markers:
            try:
                self.plot.removeItem(marker)
            except Exception:
                pass
        markers = []
        try:
            handles = list(roi.getHandles())
        except Exception:
            handles = []
        if not self.drawing_interaction_enabled:
            for handle in handles:
                try:
                    handle.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                    handle.setOpacity(0.0)
                    handle.setVisible(False)
                except Exception:
                    pass
            roi._standard_handle_markers = []
            return
        for handle in handles:
            try:
                handle.setOpacity(1.0)
            except Exception:
                pass
            pos = self._roi_handle_absolute_position(roi, handle)
            if pos is None:
                continue
            markers.append(self._make_standard_adjustment_marker(pos[0], pos[1], visible=selected))
        roi._standard_handle_markers = markers
        self._update_standard_roi_handle_markers(roi, selected=selected)

    def _update_standard_roi_handle_markers(self, roi, selected: bool | None = None):
        markers = list(getattr(roi, "_standard_handle_markers", []) or [])
        try:
            handles = list(roi.getHandles())
        except Exception:
            handles = []
        if selected is None:
            selected = bool(getattr(roi, "_standard_handles_selected", False))
        roi._standard_handles_selected = bool(selected)
        for marker, handle in zip(markers, handles):
            pos = self._roi_handle_absolute_position(roi, handle)
            if pos is not None:
                try:
                    marker.setData([pos[0]], [pos[1]])
                except Exception:
                    pass
            try:
                marker.setVisible(bool(self.drawing_interaction_enabled and selected))
            except Exception:
                pass

    @staticmethod
    def _configure_right_bottom_resize_handles(roi):
        """Text-box handles: right-middle adjusts width; bottom-middle adjusts height."""
        ChartWidget._clear_roi_handles(roi)
        try:
            right_handle = roi.addScaleHandle([1.0, 0.5], [0.0, 0.5], name="resize_right")
            bottom_handle = roi.addScaleHandle([0.5, 0.0], [0.5, 1.0], name="resize_bottom")
            for handle in (right_handle, bottom_handle):
                try:
                    handle.setZValue(30)
                except Exception:
                    pass
        except Exception:
            pass

    @staticmethod
    def _configure_four_side_resize_handles(roi):
        """Rectangle handles: one midpoint on each edge, with no corner handles.

        Each handle scales only the corresponding edge while the opposite edge
        remains anchored:
        - left-middle  -> adjusts left edge
        - right-middle -> adjusts right edge
        - top-middle   -> adjusts top edge
        - bottom-middle-> adjusts bottom edge
        """
        ChartWidget._clear_roi_handles(roi)
        try:
            left_handle = roi.addScaleHandle([0.0, 0.5], [1.0, 0.5], name="resize_left")
            right_handle = roi.addScaleHandle([1.0, 0.5], [0.0, 0.5], name="resize_right")
            top_handle = roi.addScaleHandle([0.5, 1.0], [0.5, 0.0], name="resize_top")
            bottom_handle = roi.addScaleHandle([0.5, 0.0], [0.5, 1.0], name="resize_bottom")
            for handle in (left_handle, right_handle, top_handle, bottom_handle):
                try:
                    handle.setZValue(30)
                except Exception:
                    pass
        except Exception:
            pass

    def _normalize_text_drawing(self, drawing: dict) -> tuple[dict, dict]:
        style = drawing.setdefault("style", {})
        for key, value in self._default_text_style().items():
            style.setdefault(key, value)
        box = drawing.setdefault("box", {})
        default_w, default_h = self._default_text_box_size()
        try:
            width = abs(float(box.get("width", default_w)))
        except Exception:
            width = default_w
        try:
            height = abs(float(box.get("height", default_h)))
        except Exception:
            height = default_h
        box["width"] = max(width, 1e-6)
        box["height"] = max(height, 1e-9)
        return style, box

    def _text_box_geometry(self, drawing: dict) -> tuple[float, float, float, float]:
        _style, box = self._normalize_text_drawing(drawing)
        left = float(drawing.get("time", 0.0))
        top = float(drawing.get("price", 0.0))
        width = max(float(box.get("width", 1.0)), 1e-6)
        height = max(float(box.get("height", 1.0)), 1e-9)
        right = left + width
        bottom = top - height
        return left, right, bottom, top

    def _text_box_pixel_size(self, left: float, right: float, bottom: float, top: float) -> tuple[float, float]:
        try:
            a = self.plot.vb.mapViewToScene(QtCore.QPointF(left, top))
            b = self.plot.vb.mapViewToScene(QtCore.QPointF(right, bottom))
            return max(abs(float(b.x() - a.x())), 24.0), max(abs(float(b.y() - a.y())), 18.0)
        except Exception:
            return 180.0, 80.0

    @staticmethod
    def _set_roi_handles_visible(roi, visible: bool):
        try:
            for handle in roi.getHandles():
                handle.setVisible(bool(visible))
        except Exception:
            pass

    def _update_text_box_view(self, did: str, update_roi: bool = False):
        drawing = self._drawing_by_id(did)
        group = self.drawing_items.get(did)
        if drawing is None or not isinstance(group, dict):
            return
        roi = group.get("roi")
        text_item = group.get("text")
        border_item = group.get("border")
        selection_item = group.get("selection")
        background_item = group.get("background")
        if roi is None or text_item is None:
            return

        style, _box = self._normalize_text_drawing(drawing)
        left, right, bottom, top = self._text_box_geometry(drawing)
        width = max(right - left, 1e-6)
        height = max(top - bottom, 1e-9)

        if update_roi:
            try:
                roi.blockSignals(True)
                roi.setPos([left, bottom])
                roi.setSize([width, height])
            finally:
                roi.blockSignals(False)

        rect = QtCore.QRectF(left, bottom, width, height)
        if background_item is not None:
            background_item.setRect(rect)
            if bool(style.get("background_enabled", False)):
                color = QtGui.QColor(style.get("background_color", "#181c27"))
                opacity = max(0, min(100, int(style.get("background_opacity", 0))))
                color.setAlpha(int(round(opacity * 255 / 100.0)))
                background_item.setBrush(QtGui.QBrush(color))
            else:
                background_item.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))

        outline_x = [left, right, right, left, left]
        outline_y = [top, top, bottom, bottom, top]
        if border_item is not None:
            try:
                border_item.setData(outline_x, outline_y)
                if bool(style.get("border_enabled", False)):
                    border_item.setPen(pg.mkPen(style.get("border_color", "#ffffff"), width=max(1, int(style.get("border_width", 1)))))
                else:
                    border_item.setPen(pg.mkPen((255, 255, 255, 0), width=1))
            except Exception:
                pass
        if selection_item is not None:
            try:
                selection_item.setData(outline_x, outline_y)
                if did == self.selected_drawing_id:
                    selection_item.setPen(pg.mkPen("#ffff00", width=2))
                    selection_item.show()
                else:
                    selection_item.hide()
            except Exception:
                pass

        try:
            text_item.setText(str(drawing.get("text", "")), color=style.get("color", "#ffffff"))
        except Exception:
            try:
                text_item.setText(str(drawing.get("text", "")))
                text_item.setColor(style.get("color", "#ffffff"))
            except Exception:
                pass
        font = QtGui.QFont()
        font.setPointSize(max(6, int(style.get("font_size", 12))))
        font.setBold(bool(style.get("bold", False)))
        font.setItalic(bool(style.get("italic", False)))
        try:
            text_item.textItem.setFont(font)
        except Exception:
            pass
        try:
            text_item.setPos(left, top)
        except Exception:
            pass

        px_w, px_h = self._text_box_pixel_size(left, right, bottom, top)
        try:
            if bool(style.get("auto_wrap", True)):
                text_item.textItem.setTextWidth(max(px_w - 8.0, 20.0))
            else:
                text_item.textItem.setTextWidth(-1)
            # Page size gives Qt's text document a stable editing box.  Width controls wrapping;
            # height is also persisted as part of the drawing even when content is shorter.
            doc = text_item.textItem.document()
            if bool(style.get("auto_wrap", True)):
                doc.setPageSize(QtCore.QSizeF(max(px_w - 8.0, 20.0), max(px_h - 6.0, 16.0)))
        except Exception:
            pass

    def _text_region_changed(self, did: str):
        if not self.drawing_interaction_enabled:
            return
        group = self.drawing_items.get(did)
        drawing = self._drawing_by_id(did)
        if not isinstance(group, dict) or drawing is None:
            return
        roi = group.get("roi")
        if roi is None:
            return
        try:
            pos = roi.pos(); size = roi.size()
            current = (float(pos.x()), float(pos.y()), float(pos.x()+size.x()), float(pos.y()+size.y()))
        except Exception:
            return
        previous = getattr(roi, "_magnet_prev_rect", current)
        if self._ctrl_pressed() and not getattr(roi, "_magnet_guard", False):
            left, bottom, right, top = current
            pl, pb, pr, pt = previous
            width_changed = abs((right-left) - (pr-pl)) > 1e-9
            height_changed = abs((top-bottom) - (pt-pb)) > 1e-9
            roi._magnet_guard = True
            try:
                if width_changed or height_changed:
                    cursor = self._cursor_view_position()
                    if cursor is not None:
                        sx, sy = self._magnet_snap_point(*cursor)
                        if width_changed:
                            if abs(left-pl) >= abs(right-pr): left = sx
                            else: right = sx
                        if height_changed:
                            if abs(bottom-pb) >= abs(top-pt): bottom = sy
                            else: top = sy
                        if right < left: left, right = right, left
                        if top < bottom: bottom, top = top, bottom
                        roi.setPos([left, bottom])
                        roi.setSize([max(right-left, 1e-6), max(top-bottom, 1e-9)])
                else:
                    dx, dy = self._magnet_cursor_delta()
                    if abs(dx) > 0 or abs(dy) > 0:
                        roi.setPos([left + dx, bottom + dy])
            finally:
                roi._magnet_guard = False
        try:
            pos = roi.pos(); size = roi.size()
            left = float(pos.x()); bottom = float(pos.y())
            width = max(float(size.x()), 1e-6); height = max(float(size.y()), 1e-9)
            drawing["time"] = left
            drawing["price"] = bottom + height
            drawing.setdefault("box", {})["width"] = width
            drawing.setdefault("box", {})["height"] = height
            roi._magnet_prev_rect = (left, bottom, left + width, bottom + height)
            self._update_text_box_view(did, update_roi=False)
        except Exception:
            pass

    def _sync_text_box(self, roi, drawing, emit=True):
        if not self.drawing_interaction_enabled:
            return
        try:
            pos = roi.pos(); size = roi.size()
            left = float(pos.x()); bottom = float(pos.y())
            width = max(float(size.x()), 1e-6); height = max(float(size.y()), 1e-9)
            drawing["time"] = left
            drawing["price"] = bottom + height
            drawing.setdefault("box", {})["width"] = width
            drawing.setdefault("box", {})["height"] = height
            if emit:
                self.case.touch(); self.dirty.emit(); self.history_committed.emit("Move/Resize Text Box")
        except Exception:
            pass

    def _render_text_box(self, drawing: dict):
        did = drawing.get("id")
        style, _box = self._normalize_text_drawing(drawing)
        left, right, bottom, top = self._text_box_geometry(drawing)
        width = max(right-left, 1e-6); height = max(top-bottom, 1e-9)

        background = QtWidgets.QGraphicsRectItem(left, bottom, width, height)
        background.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        background.setZValue(5)
        background.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.plot.addItem(background)

        border = pg.PlotDataItem(
            x=[left, right, right, left, left],
            y=[top, top, bottom, bottom, top],
            pen=pg.mkPen((255, 255, 255, 0), width=1),
        )
        border.setZValue(6)
        border.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.plot.addItem(border)

        selection = pg.PlotDataItem(
            x=[left, right, right, left, left],
            y=[top, top, bottom, bottom, top],
            pen=pg.mkPen("#ffff00", width=2),
        )
        selection.setZValue(8)
        selection.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        selection.hide()
        self.plot.addItem(selection)

        text_item = MovableTextItem(text=str(drawing.get("text", "")), color=style.get("color", "#ffffff"), anchor=(0, 0))
        text_item.setZValue(7)
        text_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        try:
            text_item.setAcceptedMouseButtons(
                QtCore.Qt.RightButton if self.drawing_interaction_enabled else QtCore.Qt.NoButton
            )
            text_item.textItem.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        except Exception:
            pass
        text_item.context_menu_callback = (
            (lambda d=did: self._show_drawing_context_menu(d))
            if self.drawing_interaction_enabled else None
        )
        self.plot.addItem(text_item)

        # Bare ROI has no default top-right handle; only our two midpoint
        # handles are installed below.
        roi = pg.ROI(
            [left, bottom], [width, height],
            pen=pg.mkPen((255,255,0,0), width=1),
            movable=self.drawing_interaction_enabled,
        )
        try:
            roi.setHoverPen(pg.mkPen((255,255,0,0), width=1))
        except Exception:
            pass
        if self.drawing_interaction_enabled:
            self._configure_right_bottom_resize_handles(roi)
        roi.setZValue(8)
        roi.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self._set_roi_handles_visible(roi, False)
        roi._magnet_guard = False
        roi._magnet_prev_rect = (left, bottom, right, top)
        self.plot.addItem(roi)
        if self.drawing_interaction_enabled:
            self._install_standard_roi_handle_markers(roi, selected=False)

        group = {"type": "text", "roi": roi, "text": text_item, "border": border, "selection": selection, "background": background}
        self.drawing_items[did] = group
        for item in (roi, text_item, border, selection, background):
            self._register_drawing_hit_item(did, item)
            try: item.setToolTip(f"text / {did}")
            except Exception: pass

        if self.drawing_interaction_enabled:
            roi.sigRegionChanged.connect(lambda _=None, d=did: self._text_region_changed(d))
            roi.sigRegionChanged.connect(lambda _=None, obj=roi: self._update_standard_roi_handle_markers(obj))
            roi.sigRegionChangeFinished.connect(lambda obj=roi, d=drawing: self._sync_text_box(obj, d))
        self._update_text_box_view(did, update_roi=False)

    def _select_drawing_id(self, did: str | None):
        if not self.drawing_interaction_enabled:
            self.selected_drawing_id = None
            self._refresh_selection_visuals()
            return
        self.selected_drawing_id = did
        self._refresh_selection_visuals()
        if did is not None:
            try:
                self.graphics.setFocus(QtCore.Qt.MouseFocusReason)
            except Exception:
                self.setFocus(QtCore.Qt.MouseFocusReason)

    def _refresh_selection_visuals(self):
        if self.case is None:
            return
        by_id = {d.get("id"): d for d in self.case.drawings}
        for did, item in self.drawing_items.items():
            drawing = by_id.get(did, {})
            selected = bool(self.drawing_interaction_enabled and did == self.selected_drawing_id)
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
                if dtype == "trend_line":
                    try:
                        self._set_roi_handles_visible(item, selected)
                        self._update_standard_roi_handle_markers(item, selected=selected)
                    except Exception:
                        pass
            elif dtype == "text" and isinstance(item, dict):
                roi = item.get("roi")
                if roi is not None:
                    try:
                        roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton if selected else QtCore.Qt.NoButton)
                        roi.setPen(pg.mkPen((255, 255, 0, 0), width=1))
                        self._set_roi_handles_visible(roi, selected)
                        self._update_standard_roi_handle_markers(roi, selected=selected)
                    except Exception:
                        pass
                # Selection is represented by the yellow resize box, never by changing text color.
                self._update_text_box_view(did, update_roi=False)
            elif dtype == "rectangle" and isinstance(item, dict):
                roi = item.get("roi")
                if roi is not None:
                    try:
                        roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton if selected else QtCore.Qt.NoButton)
                        roi.setPen(pg.mkPen((255, 255, 255, 0), width=1))
                        self._set_roi_handles_visible(roi, selected)
                        self._update_standard_roi_handle_markers(roi, selected=selected)
                    except Exception:
                        pass
                self._update_rectangle_fill(item)
            elif dtype == "fibonacci" and isinstance(item, dict):
                self._update_fibo_view_geometry(did, update_box=False)

    def _show_drawing_context_menu(self, did: str):
        if not self.drawing_interaction_enabled:
            return
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        dtype = drawing.get("type")
        menu = QtWidgets.QMenu(self)

        # Settings / quick edit actions first.
        if dtype == "text":
            settings_action = menu.addAction("文字設定")
            settings_action.triggered.connect(lambda: self._open_text_settings(did))
            color_action = menu.addAction("改文字顏色")
            color_action.triggered.connect(lambda: self._change_text_color(did))
            size_action = menu.addAction("改字體大小")
            size_action.triggered.connect(lambda: self._change_text_size(did))
        elif dtype == "rectangle":
            settings_action = menu.addAction("長方形設定")
            settings_action.triggered.connect(lambda: self._open_rectangle_settings(did))
        elif dtype == "fibonacci":
            settings_action = menu.addAction("Fibo 設定")
            settings_action.triggered.connect(lambda: self._open_fibo_settings(did))
            color_action = menu.addAction("改顏色")
            color_action.triggered.connect(lambda: self._change_fibo_color(did))
        else:
            settings_action = menu.addAction("線條設定")
            settings_action.triggered.connect(lambda: self._open_line_settings(did))
            color_action = menu.addAction("改顏色")
            color_action.triggered.connect(lambda: self._change_line_color(did))

        # Template submenu is available for every supported drawing type.
        template_menu = menu.addMenu("模板")
        self._populate_template_menu(template_menu, did, dtype)

        # Delete is intentionally always the last item for every drawing type.
        menu.addSeparator()
        delete_action = menu.addAction("刪除")
        delete_action.triggered.connect(lambda: self._delete_drawing_by_id(did))
        menu.exec(QtGui.QCursor.pos())

    def _populate_template_menu(self, submenu: QtWidgets.QMenu, did: str, dtype: str):
        category = self.template_repo.category_for_drawing_type(dtype)
        templates = self.template_repo.list(category)
        if not templates:
            empty = submenu.addAction("（尚無模板）")
            empty.setEnabled(False)
            return
        for template in templates:
            name = str(template.get("name", "未命名模板"))
            action = submenu.addAction(name)
            action.triggered.connect(
                lambda _checked=False, d=did, t=deepcopy(template): self._apply_drawing_template(d, t)
            )

    def _save_template_interactive(self, category: str, payload: dict):
        name, ok = QtWidgets.QInputDialog.getText(self, "存為模板", "模板名稱:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if self.template_repo.exists(category, name):
            answer = QtWidgets.QMessageBox.question(
                self,
                "覆蓋模板",
                f'模板「{name}」已存在，是否覆蓋？',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        try:
            path = self.template_repo.save(category, name, payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "模板儲存失敗", str(exc))
            return
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"已儲存模板：{name}\n{path}")

    def _apply_drawing_template(self, did: str, template: dict):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        dtype = drawing.get("type")
        category = self.template_repo.category_for_drawing_type(dtype)
        if str(template.get("drawing_type", category)) != category:
            return

        if dtype in {"horizontal_line", "trend_line", "rectangle", "text"}:
            if isinstance(template.get("style"), dict):
                drawing["style"] = deepcopy(template["style"])
                if dtype == "text" and not bool(drawing["style"].get("auto_wrap", True)):
                    self._fit_text_box_width_to_content(drawing)
        elif dtype == "fibonacci":
            if isinstance(template.get("levels"), list):
                drawing["levels"] = deepcopy(template["levels"])
            if isinstance(template.get("style"), dict):
                drawing["style"] = deepcopy(template["style"])
        self._commit_drawing_change()

    def _open_line_settings(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style = drawing.setdefault("style", {})
        dialog = LineSettingsDialog(
            style.get("color", "#ffffff"),
            style.get("line_style", "solid"),
            int(style.get("width", 2)),
            save_template_callback=lambda payload: self._save_template_interactive("line", payload),
            parent=self,
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

    def _open_text_settings(self, did: str):
        drawing = self._drawing_by_id(did)
        if drawing is None:
            return
        style, _box = self._normalize_text_drawing(drawing)
        dialog = TextSettingsDialog(
            text=str(drawing.get("text", "")),
            text_color=style.get("color", "#ffffff"),
            font_size=int(style.get("font_size", 12)),
            bold=bool(style.get("bold", False)),
            italic=bool(style.get("italic", False)),
            border_enabled=bool(style.get("border_enabled", False)),
            border_color=style.get("border_color", "#ffffff"),
            border_width=int(style.get("border_width", 1)),
            background_enabled=bool(style.get("background_enabled", False)),
            background_color=style.get("background_color", "#181c27"),
            background_opacity=int(style.get("background_opacity", 0)),
            auto_wrap=bool(style.get("auto_wrap", True)),
            save_template_callback=lambda payload: self._save_template_interactive("text", payload),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        text, new_style = dialog.values()
        drawing["text"] = text
        drawing["style"] = new_style
        if not bool(new_style.get("auto_wrap", True)):
            self._fit_text_box_width_to_content(drawing)
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
            save_template_callback=lambda payload: self._save_template_interactive("rectangle", payload),
            parent=self,
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
        dialog = FiboSettingsDialog(
            drawing.get("levels", []),
            drawing.get("style", {}),
            save_template_callback=lambda payload: self._save_template_interactive("fibonacci", payload),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            levels, style = dialog.values()
            drawing["levels"] = levels
            drawing["style"] = style
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
        if not self.drawing_interaction_enabled:
            return
        if self.case is None:
            return
        self.case.drawings = [d for d in self.case.drawings if d.get("id") != did]
        if self.selected_drawing_id == did:
            self.selected_drawing_id = None
        self._commit_drawing_change("Delete Drawing")

    def _commit_drawing_change(self, label: str = "Edit Drawing"):
        if not self.drawing_interaction_enabled:
            return
        if self.case is None:
            return
        self.case.touch()
        self.dirty.emit()
        self.history_committed.emit(label)
        self.render(reset_x=False)

    def rebuild_drawings(self):
        self.drawing_items.clear()
        self.drawing_hit_items.clear()
        self._drawing_hit_objects.clear()
        self._render_drawing_items()

    def _register_drawing_hit_item(self, did: str, item):
        if not self.drawing_interaction_enabled:
            return
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

    def _fibo_anchor_size(self, ymin: float, ymax: float) -> tuple[float, float]:
        """Return an approximately 16px square hit target in current view coordinates."""
        try:
            x_per_px, y_per_px = self.plot.vb.viewPixelSize()
            width = max(abs(float(x_per_px)) * 16.0, 1e-6)
            height = max(abs(float(y_per_px)) * 16.0, 1e-9)
            return width, height
        except Exception:
            pass
        y_span = max(float(ymax) - float(ymin), max(abs(float(ymax)), 1.0) * 1e-5)
        return max(float(timeframe_seconds(self.timeframe_combo.currentText())) * 0.25, 1e-6), max(y_span * 0.035, 1e-9)

    @staticmethod
    def _set_fibo_anchor_roi(roi, x: float, y: float, width: float, height: float):
        roi.blockSignals(True)
        try:
            roi.setPos([float(x) - float(width) * 0.5, float(y) - float(height) * 0.5])
            roi.setSize([max(float(width), 1e-6), max(float(height), 1e-9)])
            roi._fibo_expected_x = float(x)
            roi._fibo_expected_y = float(y)
        finally:
            roi.blockSignals(False)

    def _fibo_line_pen(self, drawing: dict, level_index: int, selected: bool = False):
        # Selection is represented by the two 0/1 anchor points; keep the level
        # colors/width unchanged so selecting a Fibo does not recolor the whole tool.
        levels = drawing.get("levels", []) or []
        style = drawing.get("style", {}) or {}
        base = levels[level_index].get("color", "#ffffff") if level_index < len(levels) else "#ffffff"
        width = max(1, int(style.get("width", 2)))
        qt_style = self._qt_line_style(str(style.get("line_style", "solid")))
        return pg.mkPen(base, width=width, style=qt_style)

    def _update_fibo_view_geometry(self, did: str, update_box: bool = True):
        del update_box  # kept for backward-compatible call sites
        drawing = self._drawing_by_id(did)
        group = self.drawing_items.get(did)
        if drawing is None or not isinstance(group, dict):
            return
        sx, sy, ex, ey, _left, _right, ymin, ymax, ys = self._fibo_geometry(drawing)
        anchor_w, anchor_h = self._fibo_anchor_size(ymin, ymax)
        selected = bool(self.drawing_interaction_enabled and did == self.selected_drawing_id)

        lines = group.get("lines", [])
        hit_lines = group.get("hit_lines", [])
        for i, y in enumerate(ys):
            if i < len(lines):
                lines[i].setData([sx, ex], [y, y])
                lines[i].setPen(self._fibo_line_pen(drawing, i, selected=selected))
            if i < len(hit_lines):
                hit_lines[i].setData([sx, ex], [y, y])

        anchors = group.get("anchors", [])
        markers = group.get("anchor_markers", [])
        points = ((sx, sy), (ex, ey))
        for i, (x, y) in enumerate(points):
            if i < len(anchors):
                anchor = anchors[i]
                self._set_fibo_anchor_roi(anchor, x, y, anchor_w, anchor_h)
                anchor.setAcceptedMouseButtons(
                    QtCore.Qt.LeftButton if (self.drawing_interaction_enabled and selected) else QtCore.Qt.NoButton
                )
                anchor.setVisible(bool(self.drawing_interaction_enabled and selected))
                anchor.setZValue(27)
            if i < len(markers):
                marker = markers[i]
                marker.setData([x], [y])
                marker.setVisible(bool(selected))
                marker.setZValue(26)

        group["anchor_size"] = (anchor_w, anchor_h)

    def _sync_fibo_from_anchor(self, did: str, anchor_index: int):
        if not self.drawing_interaction_enabled:
            return
        drawing = self._drawing_by_id(did)
        group = self.drawing_items.get(did)
        if drawing is None or not isinstance(group, dict) or group.get("syncing"):
            return
        anchors = group.get("anchors", [])
        if anchor_index < 0 or anchor_index >= len(anchors):
            return
        anchor = anchors[anchor_index]

        group["syncing"] = True
        try:
            pos = anchor.pos(); size = anchor.size()
            new_x = float(pos.x() + size.x() * 0.5)
            new_y = float(pos.y() + size.y() * 0.5)
            if self._ctrl_pressed():
                new_x, new_y = self._magnet_snap_point(new_x, new_y)

            target = drawing["start"] if anchor_index == 0 else drawing["end"]
            target["time"] = float(new_x)
            target["price"] = float(new_y)

            self._update_fibo_view_geometry(did)
            self.case.touch(); self.dirty.emit()
        finally:
            group["syncing"] = False

    def _finish_fibo_change(self, did: str):
        if not self.drawing_interaction_enabled:
            return
        if self.case is None or self._drawing_by_id(did) is None:
            return
        self.case.touch()
        self.dirty.emit()
        self.history_committed.emit("Move/Resize Fibonacci")

    def _render_fibonacci(self, drawing: dict):
        did = drawing.get("id")
        sx, sy, ex, ey, _left, _right, ymin, ymax, ys = self._fibo_geometry(drawing)
        anchor_w, anchor_h = self._fibo_anchor_size(ymin, ymax)
        selected = bool(self.drawing_interaction_enabled and did == self.selected_drawing_id)

        lines = []
        hit_lines = []
        levels = drawing.get("levels", []) or []
        for i, y in enumerate(ys):
            line = pg.PlotDataItem(x=[sx, ex], y=[y, y], pen=self._fibo_line_pen(drawing, i, selected=False))
            line.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            line.setZValue(10)
            self.plot.addItem(line)
            lines.append(line)

            # Wider invisible hit line is needed only in Analyzer for selection/context menus.
            if self.drawing_interaction_enabled:
                hit = pg.PlotDataItem(x=[sx, ex], y=[y, y], pen=pg.mkPen((255, 255, 255, 0), width=12))
                hit.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                hit.setZValue(11)
                self.plot.addItem(hit)
                hit_lines.append(hit)

        # Exactly two draggable anchors define the Fibonacci geometry:
        # anchor 0 = multiplier 0 / start point, anchor 1 = multiplier 1 / end point.
        # The ROI itself is an invisible hit target; ScatterPlotItem supplies the
        # visible circular control point. Both are shown only while the Fibo is selected.
        anchors = []
        markers = []
        if self.drawing_interaction_enabled:
            transparent_pen = pg.mkPen((255, 255, 255, 0), width=1)
            for idx, (x, y) in enumerate(((sx, sy), (ex, ey))):
                anchor = pg.ROI(
                    [x - anchor_w * 0.5, y - anchor_h * 0.5],
                    [anchor_w, anchor_h],
                    pen=transparent_pen,
                    movable=True,
                )
                anchor._fibo_anchor_index = idx
                anchor._fibo_expected_x = float(x)
                anchor._fibo_expected_y = float(y)
                anchor.setAcceptedMouseButtons(QtCore.Qt.LeftButton if selected else QtCore.Qt.NoButton)
                anchor.setZValue(27)
                anchor.setVisible(bool(selected))
                try:
                    anchor.setHoverPen(transparent_pen)
                except Exception:
                    pass
                self.plot.addItem(anchor)
                anchors.append(anchor)

                marker = self._make_standard_adjustment_marker(x, y, visible=selected)
                marker.setZValue(26)
                markers.append(marker)

        group = {
            "type": "fibonacci",
            "lines": lines,
            "hit_lines": hit_lines,
            "anchors": anchors,
            "anchor_markers": markers,
            "syncing": False,
            "anchor_size": (anchor_w, anchor_h),
        }
        self.drawing_items[did] = group

        for item in [*anchors, *markers, *hit_lines, *lines]:
            self._register_drawing_hit_item(did, item)
            try:
                item.setToolTip(f"fibonacci / {did}")
            except Exception:
                pass

        if self.drawing_interaction_enabled:
            for i, anchor in enumerate(anchors):
                anchor.sigRegionChanged.connect(lambda _=None, d=did, idx=i: self._sync_fibo_from_anchor(d, idx))
                anchor.sigRegionChangeFinished.connect(lambda _=None, d=did: self._finish_fibo_change(d))

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
            left = float(pos.x()); bottom = float(pos.y())
            right = left + float(size.x()); top = bottom + float(size.y())
            fill.setRect(left, bottom, right - left, top - bottom)
            drawing = group.get("drawing") or {}
            style = drawing.get("style", {})
            outline_x = [left, right, right, left, left]
            outline_y = [top, top, bottom, bottom, top]
            outline = group.get("outline")
            if outline is not None:
                outline.setData(outline_x, outline_y)
                outline.setPen(pg.mkPen(style.get("border_color", "#ffffff"), width=max(1, int(style.get("width", 2)))))
            selection = group.get("selection")
            if selection is not None:
                selection.setData(outline_x, outline_y)
                if drawing.get("id") == self.selected_drawing_id:
                    selection.setPen(pg.mkPen("#ffff00", width=max(2, int(style.get("width", 2)) + 1)))
                    selection.show()
                else:
                    selection.hide()
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

        outline = pg.PlotDataItem(
            x=[left, right, right, left, left],
            y=[top, top, bottom, bottom, top],
            pen=pg.mkPen(border, width=int(style.get("width", 2))),
        )
        outline.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        outline.setZValue(4)
        self.plot.addItem(outline)

        selection = pg.PlotDataItem(
            x=[left, right, right, left, left],
            y=[top, top, bottom, bottom, top],
            pen=pg.mkPen("#ffff00", width=max(2, int(style.get("width", 2)) + 1)),
        )
        selection.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        selection.setZValue(5)
        selection.hide()
        self.plot.addItem(selection)

        # Bare ROI prevents pyqtgraph's built-in corner handle from appearing.
        # Rectangle uses exactly four edge-midpoint resize handles.
        roi = pg.ROI(
            [left, bottom], [right - left, top - bottom],
            pen=pg.mkPen((255, 255, 255, 0), width=1),
            movable=self.drawing_interaction_enabled,
        )
        try:
            roi.setHoverPen(pg.mkPen((255, 255, 255, 0), width=1))
        except Exception:
            pass
        if not self.drawing_interaction_enabled:
            roi.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        if self.drawing_interaction_enabled:
            self._configure_four_side_resize_handles(roi)
        self._set_roi_handles_visible(roi, False)
        roi.setZValue(6)
        self.plot.addItem(roi)
        if self.drawing_interaction_enabled:
            self._install_standard_roi_handle_markers(roi, selected=False)
        group = {"type": "rectangle", "roi": roi, "fill": fill_item, "outline": outline, "selection": selection, "drawing": drawing}
        self.drawing_items[did] = group
        self._register_drawing_hit_item(did, roi)
        self._register_drawing_hit_item(did, fill_item)
        self._register_drawing_hit_item(did, outline)
        self._register_drawing_hit_item(did, selection)
        roi._magnet_guard = False
        roi._magnet_prev_rect = (left, bottom, right, top)
        if self.drawing_interaction_enabled:
            roi.sigRegionChanged.connect(lambda _=None, d=did: self._rectangle_region_changed(d))
            roi.sigRegionChanged.connect(lambda _=None, obj=roi: self._update_standard_roi_handle_markers(obj))
            roi.sigRegionChangeFinished.connect(lambda obj=roi, d=drawing: self._sync_rectangle(obj, d))

    def _render_drawing_items(self):
        self.drawing_items.clear()
        self.drawing_hit_items.clear()
        self._drawing_hit_objects.clear()
        if self.case is None or not self.show_all_drawings:
            return
        for drawing in self.case.drawings:
            dtype = drawing.get("type")
            did = drawing.get("id")
            style = drawing.get("style", {})
            if dtype == "horizontal_line":
                item = pg.InfiniteLine(
                    pos=float(drawing["price"]), angle=0,
                    movable=self.drawing_interaction_enabled,
                    pen=self._pen_from_style(style, selected=False),
                )
                item._magnet_guard = False
                if self.drawing_interaction_enabled:
                    item.sigPositionChanged.connect(lambda _=None, obj=item: self._magnetize_hline(obj))
                    item.sigPositionChangeFinished.connect(lambda obj=item, d=drawing: self._sync_hline(obj, d))
                else:
                    item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                self.plot.addItem(item)
            elif dtype == "trend_line":
                p1, p2 = drawing["points"]
                item = pg.LineSegmentROI(
                    [(p1["time"], p1["price"]), (p2["time"], p2["price"])],
                    pen=self._pen_from_style(style, selected=False),
                    movable=self.drawing_interaction_enabled,
                )
                item._shift_prev_points = [(float(p1["time"]), float(p1["price"])), (float(p2["time"]), float(p2["price"]))]
                item._shift_constraint_guard = False
                if self.drawing_interaction_enabled:
                    item.sigRegionChanged.connect(lambda _=None, obj=item: self._enforce_trend_shift_constraint(obj))
                    item.sigRegionChanged.connect(lambda _=None, obj=item: self._update_standard_roi_handle_markers(obj))
                    item.sigRegionChangeFinished.connect(lambda obj=item, d=drawing: self._sync_trend(obj, d))
                else:
                    item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                    try:
                        for handle in item.getHandles():
                            handle.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                            handle.setOpacity(0.0)
                    except Exception:
                        pass
                self.plot.addItem(item)
                self._set_roi_handles_visible(item, False)
                if self.drawing_interaction_enabled:
                    self._install_standard_roi_handle_markers(item, selected=False)
            elif dtype == "text":
                self._render_text_box(drawing)
                continue
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
        if not self.drawing_interaction_enabled:
            return
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
                roi = item.get("roi") if isinstance(item, dict) else None
                if roi is not None:
                    self._sync_text_box(roi, d, emit=False)
            elif d.get("type") == "rectangle":
                roi = item.get("roi") if isinstance(item, dict) else None
                if roi is not None:
                    self._sync_rectangle(roi, d, emit=False)
            elif d.get("type") == "fibonacci":
                # Fibo domain data is updated live from the handle-free line control strips.
                pass

    def _magnetize_hline(self, item):
        if not self._ctrl_pressed() or getattr(item, "_magnet_guard", False):
            return
        cursor = self._cursor_view_position()
        if cursor is None:
            return
        try:
            _sx, sy = self._magnet_snap_point(cursor[0], float(item.value()))
            if abs(float(item.value()) - sy) < 1e-12:
                return
            item._magnet_guard = True
            item.setValue(float(sy))
        finally:
            item._magnet_guard = False

    def _rectangle_region_changed(self, did: str):
        if not self.drawing_interaction_enabled:
            return
        group = self.drawing_items.get(did)
        if not isinstance(group, dict):
            return
        roi = group.get("roi")
        if roi is None:
            return
        self._update_rectangle_fill(group)
        try:
            pos = roi.pos(); size = roi.size()
            current = (float(pos.x()), float(pos.y()), float(pos.x()+size.x()), float(pos.y()+size.y()))
        except Exception:
            return
        previous = getattr(roi, "_magnet_prev_rect", current)
        if not self._ctrl_pressed() or getattr(roi, "_magnet_guard", False):
            roi._magnet_prev_rect = current
            return

        left, bottom, right, top = current
        pl, pb, pr, pt = previous
        width_changed = abs((right-left) - (pr-pl)) > 1e-9
        height_changed = abs((top-bottom) - (pt-pb)) > 1e-9
        roi._magnet_guard = True
        try:
            if width_changed or height_changed:
                cursor = self._cursor_view_position()
                if cursor is not None:
                    sx, sy = self._magnet_snap_point(*cursor)
                    if width_changed:
                        if abs(left-pl) >= abs(right-pr):
                            left = sx
                        else:
                            right = sx
                    if height_changed:
                        if abs(bottom-pb) >= abs(top-pt):
                            bottom = sy
                        else:
                            top = sy
                    if right < left:
                        left, right = right, left
                    if top < bottom:
                        bottom, top = top, bottom
                    roi.setPos([left, bottom])
                    roi.setSize([max(right-left, 1e-6), max(top-bottom, 1e-9)])
            else:
                dx, dy = self._magnet_cursor_delta()
                if abs(dx) > 0 or abs(dy) > 0:
                    roi.setPos([left + dx, bottom + dy])
            pos = roi.pos(); size = roi.size()
            roi._magnet_prev_rect = (float(pos.x()), float(pos.y()), float(pos.x()+size.x()), float(pos.y()+size.y()))
            self._update_rectangle_fill(group)
        finally:
            roi._magnet_guard = False

    def _sync_hline(self, item, drawing):
        if not self.drawing_interaction_enabled:
            return
        drawing["price"] = float(item.value())
        self.case.touch(); self.dirty.emit(); self.history_committed.emit("Move Drawing")

    def _sync_trend(self, item, drawing, emit=True):
        if not self.drawing_interaction_enabled:
            return
        try:
            pts = self._line_state_points(item)
            if len(pts) != 2:
                return
            drawing["points"] = [
                {"time": float(pts[0][0]), "price": float(pts[0][1])},
                {"time": float(pts[1][0]), "price": float(pts[1][1])},
            ]
            if emit:
                self.case.touch(); self.dirty.emit(); self.history_committed.emit("Move/Resize Drawing")
        except Exception:
            pass

    def _sync_rectangle(self, item, drawing, emit=True):
        if not self.drawing_interaction_enabled:
            return
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
                self.case.touch(); self.dirty.emit(); self.history_committed.emit("Move/Resize Drawing")
        except Exception:
            pass

    def _sync_text(self, item, drawing):
        if not self.drawing_interaction_enabled:
            return
        try:
            pos = item.pos()
            drawing["time"] = float(pos.x())
            drawing["price"] = float(pos.y())
            self.case.touch(); self.dirty.emit(); self.history_committed.emit("Move Text")
        except Exception:
            pass

    def _clipboard_offset_delta(self, paste_count: int) -> tuple[float, float]:
        """Return a small screen-consistent right/down paste offset in view coordinates."""
        count = max(int(paste_count), 1)
        try:
            x_per_px, y_per_px = self.plot.vb.viewPixelSize()
            dx = abs(float(x_per_px)) * 12.0 * count
            # Screen-down corresponds to a lower price in the chart's view coordinates.
            dy = -abs(float(y_per_px)) * 12.0 * count
            if not np.isfinite(dx) or not np.isfinite(dy):
                raise ValueError("non-finite pixel size")
            return dx, dy
        except Exception:
            tf_step = max(float(timeframe_seconds(self.timeframe_combo.currentText())), 1.0)
            return tf_step * count, -1.0 * count

    def copy_selected_drawing(self) -> bool:
        """Copy the selected Drawing Domain object into the internal drawing clipboard."""
        if not self.drawing_interaction_enabled:
            return False
        if self.case is None or self.selected_drawing_id is None:
            return False
        # Persist any current drag/resize state before taking the copy.
        self.sync_all_drawings_from_view()
        drawing = self._drawing_by_id(self.selected_drawing_id)
        if drawing is None:
            return False
        self._drawing_clipboard = deepcopy(drawing)
        self._clipboard_paste_count = 0
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "已複製 Drawing  (Ctrl+V 貼上)")
        return True

    def paste_copied_drawing(self) -> bool:
        """Paste a cloned drawing with a new id and a small visible offset."""
        if not self.drawing_interaction_enabled:
            return False
        if self.case is None or self._drawing_clipboard is None:
            return False
        self._clipboard_paste_count += 1
        dx, dy = self._clipboard_offset_delta(self._clipboard_paste_count)
        clone = clone_drawing_with_offset(self._drawing_clipboard, dx, dy)
        self.case.drawings.append(clone)
        self.selected_drawing_id = str(clone.get("id"))
        self.case.touch()
        self.dirty.emit()
        self.history_committed.emit("Paste Drawing")
        self.render(reset_x=False)
        try:
            self.graphics.setFocus(QtCore.Qt.MouseFocusReason)
        except Exception:
            pass
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "已貼上 Drawing")
        return True

    def delete_selected_drawing(self):
        if not self.drawing_interaction_enabled:
            return
        if self.case is None or self.selected_drawing_id is None:
            return
        self._delete_drawing_by_id(self.selected_drawing_id)
