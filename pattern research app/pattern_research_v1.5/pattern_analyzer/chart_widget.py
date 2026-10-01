from __future__ import annotations

import math
import uuid
from copy import deepcopy
from typing import Callable
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.aggregation import aggregate_visible_bars, timeframe_seconds
from shared_core.drawing_clipboard import clone_drawing_with_offset
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
    def __init__(self, levels: list[dict], style: dict | None = None,
                 save_template_callback: Callable[[dict], None] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fibo 設定")
        self.resize(445, 360)
        self._save_template_callback = save_template_callback
        self._style = deepcopy(style or {"width": 2, "line_style": "solid"})
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

    def _save_template(self):
        if self._save_template_callback is None:
            return
        try:
            levels = self.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Fibo 設定", str(exc))
            return
        self._save_template_callback({"levels": levels, "style": deepcopy(self._style)})


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
        self._drawing_clipboard: dict | None = None
        self._clipboard_paste_count = 0
        self.template_repo = DrawingTemplateRepository()
        self.auto_all_mode = False
        self._syncing_auto_all = False
        self.order_events: list[dict] = []
        self.show_previous_rth = False

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
        self.graphics.setFocusPolicy(QtCore.Qt.StrongFocus)

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

        self._render_reference_levels()
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
        except Exception:
            return

        pen = pg.mkPen((255, 255, 255, 185), width=1, style=QtCore.Qt.DashLine)
        high_line = pg.InfiniteLine(pos=high, angle=0, movable=False, pen=pen)
        low_line = pg.InfiniteLine(pos=low, angle=0, movable=False, pen=pen)
        high_line.setZValue(2)
        low_line.setZValue(2)
        high_line.setToolTip(f"Previous RTH High: {high:.2f}")
        low_line.setToolTip(f"Previous RTH Low: {low:.2f}")
        self.plot.addItem(high_line)
        self.plot.addItem(low_line)

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

        # Keep Fibonacci 0/1 drag hit-zones roughly constant in screen pixels.
        # This prevents the controls from becoming nearly impossible to grab after zooming.
        for did, group in list(self.drawing_items.items()):
            if isinstance(group, dict) and group.get("type") == "fibonacci":
                self._update_fibo_view_geometry(did, update_box=False)
            elif isinstance(group, dict) and group.get("type") == "text":
                # Text wrapping is based on the box width in screen pixels, so refresh it after zoom/pan.
                self._update_text_box_view(did, update_roi=False)

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

        # TradingView-like crosshair: X snaps to nearest revealed candle. Holding Ctrl also snaps Y to nearest OHLC.
        if not self._last_bars.empty:
            ts = self._last_bars["timestamp"].to_numpy(dtype=float)
            x = float(ts[int(np.argmin(np.abs(ts - x)))])
        if self._ctrl_pressed():
            x, y = self._magnet_snap_point(x, y)
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
        try:
            pts = item.getState().get("points", [])
            if len(pts) >= 2:
                return [(float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))]
        except Exception:
            pass
        return []

    def _set_line_roi_points(self, item, points: list[tuple[float, float]]):
        """Best-effort point setter compatible with pyqtgraph LineSegmentROI/PolyLineROI."""
        qpts = [QtCore.QPointF(float(x), float(y)) for x, y in points]
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
            self.case.touch(); self.dirty.emit(); self.rebuild_drawings(); self.render(False)
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
        try:
            pos = roi.pos(); size = roi.size()
            left = float(pos.x()); bottom = float(pos.y())
            width = max(float(size.x()), 1e-6); height = max(float(size.y()), 1e-9)
            drawing["time"] = left
            drawing["price"] = bottom + height
            drawing.setdefault("box", {})["width"] = width
            drawing.setdefault("box", {})["height"] = height
            if emit:
                self.case.touch(); self.dirty.emit()
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
            text_item.setAcceptedMouseButtons(QtCore.Qt.RightButton)
            text_item.textItem.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        except Exception:
            pass
        text_item.context_menu_callback = lambda d=did: self._show_drawing_context_menu(d)
        self.plot.addItem(text_item)

        # Bare ROI has no default top-right handle; only our two midpoint
        # handles are installed below.
        roi = pg.ROI([left, bottom], [width, height], pen=pg.mkPen((255,255,0,0), width=1), movable=True)
        try:
            roi.setHoverPen(pg.mkPen((255,255,0,0), width=1))
        except Exception:
            pass
        self._configure_right_bottom_resize_handles(roi)
        roi.setZValue(8)
        roi.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self._set_roi_handles_visible(roi, False)
        roi._magnet_guard = False
        roi._magnet_prev_rect = (left, bottom, right, top)
        self.plot.addItem(roi)

        group = {"type": "text", "roi": roi, "text": text_item, "border": border, "selection": selection, "background": background}
        self.drawing_items[did] = group
        for item in (roi, text_item, border, selection, background):
            self._register_drawing_hit_item(did, item)
            try: item.setToolTip(f"text / {did}")
            except Exception: pass

        roi.sigRegionChanged.connect(lambda _=None, d=did: self._text_region_changed(d))
        roi.sigRegionChangeFinished.connect(lambda obj=roi, d=drawing: self._sync_text_box(obj, d))
        self._update_text_box_view(did, update_roi=False)

    def _select_drawing_id(self, did: str | None):
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
            elif dtype == "text" and isinstance(item, dict):
                roi = item.get("roi")
                if roi is not None:
                    try:
                        roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton if selected else QtCore.Qt.NoButton)
                        roi.setPen(pg.mkPen((255, 255, 0, 0), width=1))
                        self._set_roi_handles_visible(roi, selected)
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
                    except Exception:
                        pass
                self._update_rectangle_fill(item)
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

    def _fibo_handle_height(self, ymin: float, ymax: float) -> float:
        """Return a stable ~16px vertical hit target in current view coordinates."""
        try:
            _x_per_px, y_per_px = self.plot.vb.viewPixelSize()
            px_height = abs(float(y_per_px)) * 16.0
            if px_height > 0:
                return px_height
        except Exception:
            pass
        # Fallback used before the ViewBox has a valid pixel transform.
        return max((float(ymax) - float(ymin)) * 0.03, max(abs(float(ymax)), 1.0) * 1e-5)

    def _update_fibo_view_geometry(self, did: str, update_box: bool = True):
        drawing = self._drawing_by_id(did)
        group = self.drawing_items.get(did)
        if drawing is None or not isinstance(group, dict):
            return
        sx, sy, ex, ey, left, right, ymin, ymax, ys = self._fibo_geometry(drawing)
        tick = max(float(timeframe_seconds(self.timeframe_combo.currentText())), 1.0)
        h = self._fibo_handle_height(ymin, ymax)
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
            if self._ctrl_pressed():
                width_changed = abs((curr_right-curr_left) - (float(last_right)-float(last_left))) > 1e-9
                if width_changed:
                    cursor = self._cursor_view_position()
                    if cursor is not None:
                        sx, _sy = self._magnet_snap_point(*cursor)
                        if abs(curr_left-float(last_left)) >= abs(curr_right-float(last_right)):
                            if start_is_left:
                                drawing["start"]["time"] = sx
                            else:
                                drawing["end"]["time"] = sx
                        else:
                            if start_is_left:
                                drawing["end"]["time"] = sx
                            else:
                                drawing["start"]["time"] = sx
                else:
                    dx, mdy = self._magnet_cursor_delta()
                    drawing["start"]["time"] = float(drawing["start"]["time"]) + dx
                    drawing["end"]["time"] = float(drawing["end"]["time"]) + dx
                    drawing["start"]["price"] = float(drawing["start"]["price"]) + mdy
                    drawing["end"]["price"] = float(drawing["end"]["price"]) + mdy
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
            new_start = float(sp.y() + ss.y() * 0.5)
            new_end = float(ep.y() + es.y() * 0.5)
            old_start = float(drawing["start"]["price"])
            old_end = float(drawing["end"]["price"])
            if self._ctrl_pressed():
                cursor = self._cursor_view_position()
                if cursor is not None:
                    _sx, sy = self._magnet_snap_point(*cursor)
                    if abs(new_start-old_start) >= abs(new_end-old_end):
                        new_start = sy
                    else:
                        new_end = sy
            drawing["start"]["price"] = new_start
            drawing["end"]["price"] = new_end
            self._update_fibo_view_geometry(did, update_box=True)
            self.case.touch(); self.dirty.emit()
        finally:
            group["syncing"] = False

    def _render_fibonacci(self, drawing: dict):
        did = drawing.get("id")
        sx, sy, ex, ey, left, right, ymin, ymax, ys = self._fibo_geometry(drawing)
        tick = max(float(timeframe_seconds(self.timeframe_combo.currentText())), 1.0)
        handle_h = self._fibo_handle_height(ymin, ymax)
        handle_left, handle_right = left - 0.5 * tick, right + 0.5 * tick

        box = pg.RectROI([left, ymin], [right - left, ymax - ymin], pen=pg.mkPen((0, 0, 0, 0)), movable=True)
        try:
            box.setHoverPen(pg.mkPen((255, 255, 0, 100), width=1))
        except Exception:
            pass
        self.plot.addItem(box)

        # The visible Fibo level stays thin; these ROIs are deliberately much taller
        # invisible hit-zones so the 0/1 controls remain easy to grab at any zoom.
        transparent_pen = pg.mkPen((255, 255, 255, 0), width=1)
        start_handle = pg.RectROI(
            [handle_left, sy - handle_h * 0.5], [handle_right - handle_left, handle_h],
            pen=transparent_pen, movable=True, rotatable=False, resizable=False,
        )
        end_handle = pg.RectROI(
            [handle_left, ey - handle_h * 0.5], [handle_right - handle_left, handle_h],
            pen=transparent_pen, movable=True, rotatable=False, resizable=False,
        )
        for handle in (start_handle, end_handle):
            handle.setZValue(20)
            handle.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
            try:
                handle.setHoverPen(pg.mkPen((255, 235, 59, 210), width=2))
            except Exception:
                pass
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
            pen=pg.mkPen((255, 255, 255, 0), width=1), movable=True,
        )
        try:
            roi.setHoverPen(pg.mkPen((255, 255, 255, 0), width=1))
        except Exception:
            pass
        self._configure_four_side_resize_handles(roi)
        self._set_roi_handles_visible(roi, False)
        roi.setZValue(6)
        self.plot.addItem(roi)
        group = {"type": "rectangle", "roi": roi, "fill": fill_item, "outline": outline, "selection": selection, "drawing": drawing}
        self.drawing_items[did] = group
        self._register_drawing_hit_item(did, roi)
        self._register_drawing_hit_item(did, fill_item)
        self._register_drawing_hit_item(did, outline)
        self._register_drawing_hit_item(did, selection)
        roi._magnet_guard = False
        roi._magnet_prev_rect = (left, bottom, right, top)
        roi.sigRegionChanged.connect(lambda _=None, d=did: self._rectangle_region_changed(d))
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
                item._magnet_guard = False
                item.sigPositionChanged.connect(lambda _=None, obj=item: self._magnetize_hline(obj))
                item.sigPositionChangeFinished.connect(lambda obj=item, d=drawing: self._sync_hline(obj, d))
                self.plot.addItem(item)
            elif dtype == "trend_line":
                p1, p2 = drawing["points"]
                item = pg.LineSegmentROI(
                    [(p1["time"], p1["price"]), (p2["time"], p2["price"])],
                    pen=self._pen_from_style(style, selected=False),
                )
                item._shift_prev_points = [(float(p1["time"]), float(p1["price"])), (float(p2["time"]), float(p2["price"]))]
                item._shift_constraint_guard = False
                item.sigRegionChanged.connect(lambda _=None, obj=item: self._enforce_trend_shift_constraint(obj))
                item.sigRegionChangeFinished.connect(lambda obj=item, d=drawing: self._sync_trend(obj, d))
                self.plot.addItem(item)
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
                # Fibo domain data is updated live from the box/0x/1x handles.
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
        if self.case is None or self._drawing_clipboard is None:
            return False
        self._clipboard_paste_count += 1
        dx, dy = self._clipboard_offset_delta(self._clipboard_paste_count)
        clone = clone_drawing_with_offset(self._drawing_clipboard, dx, dy)
        self.case.drawings.append(clone)
        self.selected_drawing_id = str(clone.get("id"))
        self.case.touch()
        self.dirty.emit()
        self.render(reset_x=False)
        try:
            self.graphics.setFocus(QtCore.Qt.MouseFocusReason)
        except Exception:
            pass
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "已貼上 Drawing")
        return True

    def delete_selected_drawing(self):
        if self.case is None or self.selected_drawing_id is None:
            return
        self._delete_drawing_by_id(self.selected_drawing_id)
