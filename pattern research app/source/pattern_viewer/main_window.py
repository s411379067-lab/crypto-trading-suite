from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.market_data import MarketDataService
from shared_core.order_overlay import build_order_overlay
from shared_core.replay import ReplayEngine
from shared_core.repository import CaseRepository
from shared_core.models import new_id, utc_now_iso
from pattern_analyzer.main_window import APP_STYLE

from .filtering import (
    ViewerCaseEntry,
    collect_patterns,
    filter_case_entries,
    save_case_patterns,
    scan_case_entries,
)
from .read_only_chart import ReadOnlyChartWidget


class PatternViewerWindow(QtWidgets.QMainWindow):
    """Pattern library browser with chart-read-only viewing.

    Drawings, notes, orders and chart research state are never editable here.
    v2.3 intentionally permits one narrow write path: ``patterns[]`` can be
    added/renamed/deleted from the left sidebar via an atomic pattern-only saver.
    """

    def __init__(self, case_root: str | Path = "."):
        super().__init__()
        self.setWindowTitle("Pattern Viewer v2.8")
        self.resize(1600, 920)
        self.setStyleSheet(APP_STYLE)

        self.repo = CaseRepository()
        self.market = MarketDataService()
        self.case_root = Path(case_root).expanduser().resolve()
        self.entries: list[ViewerCaseEntry] = []
        self.filtered_entries: list[ViewerCaseEntry] = []
        self.current_case = None
        self.current_case_path: Path | None = None
        self.current_replay = None
        self.current_raw = pd.DataFrame()
        self._updating_case_list = False

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter, 1)

        self.sidebar = self._build_sidebar()
        self.chart = ReadOnlyChartWidget()
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.chart)
        splitter.setSizes([390, 1210])

        status = QtWidgets.QWidget()
        sl = QtWidgets.QHBoxLayout(status)
        sl.setContentsMargins(6, 3, 6, 3)
        self.case_status = QtWidgets.QLabel("尚未載入 Case")
        self.range_status = QtWidgets.QLabel("")
        self.readonly_badge = QtWidgets.QLabel("CHART READ ONLY • PATTERN EDITABLE")
        self.readonly_badge.setStyleSheet(
            "background-color:#24344d; border:1px solid #5d789f; border-radius:3px; "
            "color:#dbe9ff; padding:3px 8px; font-weight:700;"
        )
        sl.addWidget(self.case_status)
        sl.addSpacing(16)
        sl.addWidget(self.range_status)
        sl.addStretch(1)
        sl.addWidget(self.readonly_badge)
        outer.addWidget(status)

        self.refresh_cases(select_first=True)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(320)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(7)

        title = QtWidgets.QLabel("Pattern Library")
        title.setStyleSheet("font-size:14pt; font-weight:700; color:#e9eef8;")
        layout.addWidget(title)

        root_row = QtWidgets.QHBoxLayout()
        self.root_edit = QtWidgets.QLineEdit(str(self.case_root))
        self.root_edit.setReadOnly(True)
        self.btn_folder = QtWidgets.QPushButton("Folder")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_folder.setFixedWidth(68)
        self.btn_refresh.setFixedWidth(72)
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(self.btn_folder)
        root_row.addWidget(self.btn_refresh)
        layout.addLayout(root_row)

        filter_box = QtWidgets.QGroupBox("Pattern Filter")
        fl = QtWidgets.QVBoxLayout(filter_box)
        fl.setSpacing(5)

        self.pattern_search = QtWidgets.QLineEdit()
        self.pattern_search.setPlaceholderText("搜尋 Pattern 名稱…")
        fl.addWidget(self.pattern_search)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("符合方式"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("ANY — 任一 Pattern", "ANY")
        self.mode_combo.addItem("ALL — 所有 Pattern", "ALL")
        mode_row.addWidget(self.mode_combo, 1)
        self.btn_clear_patterns = QtWidgets.QPushButton("清除")
        self.btn_clear_patterns.setFixedWidth(58)
        mode_row.addWidget(self.btn_clear_patterns)
        fl.addLayout(mode_row)

        self.pattern_list = QtWidgets.QListWidget()
        self.pattern_list.setMinimumHeight(160)
        self.pattern_list.setMaximumHeight(240)
        fl.addWidget(self.pattern_list)
        layout.addWidget(filter_box)

        view_box = QtWidgets.QGroupBox("Viewer Overlay")
        vl = QtWidgets.QHBoxLayout(view_box)
        self.show_notes_checkbox = QtWidgets.QCheckBox("全部紀錄")
        self.show_notes_checkbox.setChecked(True)
        self.show_rth_checkbox = QtWidgets.QCheckBox("RTH H/L/C")
        self.show_rth_checkbox.setChecked(True)
        self.show_orders_checkbox = QtWidgets.QCheckBox("全部 Order")
        self.show_orders_checkbox.setChecked(False)
        self.show_orders_checkbox.setToolTip("顯示已成交 Order 的紅綠三角形，以及完成交易的 Open → Close PnL 點線")
        vl.addWidget(self.show_notes_checkbox)
        vl.addWidget(self.show_rth_checkbox)
        vl.addWidget(self.show_orders_checkbox)
        vl.addStretch(1)
        layout.addWidget(view_box)

        self.case_count_label = QtWidgets.QLabel("Cases: 0")
        self.case_count_label.setStyleSheet("font-weight:700; color:#cfd7e6;")
        layout.addWidget(self.case_count_label)

        self.case_list = QtWidgets.QListWidget()
        self.case_list.setAlternatingRowColors(True)
        # Explicit dark palette prevents Windows/Qt native alternate rows from
        # falling back to a light system color while the app still uses light text.
        case_palette = self.case_list.palette()
        case_palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#0d1420"))
        case_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#141e2d"))
        case_palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#f3f6fb"))
        case_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2b4a73"))
        case_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
        self.case_list.setPalette(case_palette)
        self.case_list.setStyleSheet(
            "QListWidget { background:#0d1420; alternate-background-color:#141e2d; "
            "color:#f3f6fb; border:1px solid #36445d; }"
            "QListWidget::item { color:#f3f6fb; padding:3px 5px; }"
            "QListWidget::item:hover { background:#1d2a3e; color:#ffffff; }"
            "QListWidget::item:selected { background:#2b4a73; color:#ffffff; }"
            "QListWidget::item:selected:!active { background:#263f61; color:#ffffff; }"
        )
        layout.addWidget(self.case_list, 1)

        detail_box = QtWidgets.QGroupBox("Selected Case")
        dl = QtWidgets.QVBoxLayout(detail_box)
        self.selected_path_label = QtWidgets.QLabel("--")
        self.selected_path_label.setWordWrap(True)
        self.selected_patterns_label = QtWidgets.QLabel("Pattern: --")
        self.selected_patterns_label.setWordWrap(True)
        dl.addWidget(self.selected_path_label)
        dl.addWidget(self.selected_patterns_label)
        layout.addWidget(detail_box)

        self.pattern_edit_box = QtWidgets.QGroupBox("Case Pattern 編輯")
        pel = QtWidgets.QVBoxLayout(self.pattern_edit_box)
        pel.setSpacing(5)

        self.case_pattern_list = QtWidgets.QListWidget()
        self.case_pattern_list.setMinimumHeight(92)
        self.case_pattern_list.setMaximumHeight(145)
        self.case_pattern_list.setStyleSheet(
            "QListWidget { background:#0d1420; color:#f3f6fb; border:1px solid #36445d; }"
            "QListWidget::item { padding:3px 5px; }"
            "QListWidget::item:selected { background:#2b4a73; color:#ffffff; }"
        )
        pel.addWidget(self.case_pattern_list)

        self.case_pattern_input = QtWidgets.QLineEdit()
        self.case_pattern_input.setPlaceholderText("輸入 Pattern…")
        pel.addWidget(self.case_pattern_input)

        edit_btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_case_pattern = QtWidgets.QPushButton("新增")
        self.btn_rename_case_pattern = QtWidgets.QPushButton("更改")
        self.btn_delete_case_pattern = QtWidgets.QPushButton("刪除")
        edit_btn_row.addWidget(self.btn_add_case_pattern)
        edit_btn_row.addWidget(self.btn_rename_case_pattern)
        edit_btn_row.addWidget(self.btn_delete_case_pattern)
        pel.addLayout(edit_btn_row)
        self.pattern_edit_box.setEnabled(False)
        layout.addWidget(self.pattern_edit_box)

        self.btn_folder.clicked.connect(self.choose_folder)
        self.btn_refresh.clicked.connect(lambda: self.refresh_cases(select_first=False))
        self.mode_combo.currentIndexChanged.connect(self.apply_filter)
        self.btn_clear_patterns.clicked.connect(self.clear_pattern_filter)
        self.pattern_search.textChanged.connect(self._apply_pattern_search_visibility)
        self.pattern_list.itemChanged.connect(self.apply_filter)
        self.case_list.currentItemChanged.connect(self._case_item_changed)
        self.show_notes_checkbox.toggled.connect(self._update_note_visibility)
        self.show_rth_checkbox.toggled.connect(self._update_rth_visibility)
        self.show_orders_checkbox.toggled.connect(self._update_order_visibility)
        self.case_pattern_list.itemSelectionChanged.connect(self._case_pattern_selection_changed)
        self.case_pattern_list.itemDoubleClicked.connect(lambda _item: self.case_pattern_input.setFocus())
        self.btn_add_case_pattern.clicked.connect(self.add_case_pattern)
        self.btn_rename_case_pattern.clicked.connect(self.rename_case_pattern)
        self.btn_delete_case_pattern.clicked.connect(self.delete_case_pattern)
        return panel

    def _refresh_case_pattern_editor(self, select_row: int | None = None):
        self.case_pattern_list.blockSignals(True)
        self.case_pattern_list.clear()
        case = self.current_case
        if case is None:
            self.pattern_edit_box.setEnabled(False)
            self.case_pattern_input.clear()
            self.case_pattern_list.blockSignals(False)
            return

        self.pattern_edit_box.setEnabled(True)
        for row, pattern in enumerate(getattr(case, "patterns", []) or []):
            if not isinstance(pattern, dict):
                continue
            text = str(pattern.get("text", "")).strip()
            if not text:
                continue
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, row)
            self.case_pattern_list.addItem(item)

        self.case_pattern_list.blockSignals(False)
        if select_row is not None and 0 <= int(select_row) < self.case_pattern_list.count():
            self.case_pattern_list.setCurrentRow(int(select_row))
        else:
            self.case_pattern_input.clear()

    def _case_pattern_selection_changed(self):
        item = self.case_pattern_list.currentItem()
        if item is None:
            return
        self.case_pattern_input.setText(item.text())
        self.case_pattern_input.selectAll()

    def _save_patterns_from_viewer(self, patterns: list[dict], action_name: str) -> bool:
        if self.current_case_path is None:
            return False
        try:
            save_case_patterns(self.current_case_path, patterns)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Pattern Save Error",
                f"{action_name}失敗：\n\n{exc}",
            )
            return False

        # Re-scan so Pattern Filter counts and Case List text immediately match disk.
        # select_first=True also moves to the next visible Case if an edit makes the
        # current Case no longer satisfy the active Pattern filter.
        self.refresh_cases(select_first=True)
        return True

    def add_case_pattern(self):
        if self.current_case is None:
            return
        text = self.case_pattern_input.text().strip()
        if not text:
            return
        now = utc_now_iso()
        patterns = [dict(x) for x in (self.current_case.patterns or []) if isinstance(x, dict)]
        patterns.append({
            "id": new_id("pattern"),
            "text": text,
            "created_at": now,
            "updated_at": now,
        })
        if self._save_patterns_from_viewer(patterns, "新增 Pattern"):
            self.case_pattern_input.clear()

    def rename_case_pattern(self):
        if self.current_case is None:
            return
        item = self.case_pattern_list.currentItem()
        if item is None:
            return
        text = self.case_pattern_input.text().strip()
        if not text:
            return
        try:
            source_row = int(item.data(QtCore.Qt.UserRole))
        except Exception:
            return
        patterns = [dict(x) for x in (self.current_case.patterns or []) if isinstance(x, dict)]
        if not (0 <= source_row < len(patterns)):
            return
        patterns[source_row]["text"] = text
        patterns[source_row]["updated_at"] = utc_now_iso()
        self._save_patterns_from_viewer(patterns, "更改 Pattern")

    def delete_case_pattern(self):
        if self.current_case is None:
            return
        item = self.case_pattern_list.currentItem()
        if item is None:
            return
        try:
            source_row = int(item.data(QtCore.Qt.UserRole))
        except Exception:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "刪除 Pattern",
            f"確定刪除「{item.text()}」？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        patterns = [dict(x) for x in (self.current_case.patterns or []) if isinstance(x, dict)]
        if not (0 <= source_row < len(patterns)):
            return
        del patterns[source_row]
        if self._save_patterns_from_viewer(patterns, "刪除 Pattern"):
            self.case_pattern_input.clear()

    def choose_folder(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇 Case Library", str(self.case_root))
        if not selected:
            return
        self.case_root = Path(selected).resolve()
        self.root_edit.setText(str(self.case_root))
        self.refresh_cases(select_first=True)

    def _selected_patterns(self) -> list[str]:
        out = []
        for i in range(self.pattern_list.count()):
            item = self.pattern_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                out.append(str(item.data(QtCore.Qt.UserRole) or item.text()).strip())
        return out

    def _rebuild_pattern_list(self, preserve: set[str] | None = None):
        preserve = preserve or set()
        counts = Counter(p for entry in self.entries for p in entry.patterns)
        patterns = collect_patterns(self.entries)
        self.pattern_list.blockSignals(True)
        self.pattern_list.clear()
        for pattern in patterns:
            item = QtWidgets.QListWidgetItem(f"{pattern}   ({counts[pattern]})")
            item.setData(QtCore.Qt.UserRole, pattern)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if pattern in preserve else QtCore.Qt.Unchecked)
            self.pattern_list.addItem(item)
        self.pattern_list.blockSignals(False)
        self._apply_pattern_search_visibility(self.pattern_search.text())

    def _apply_pattern_search_visibility(self, text: str):
        query = str(text or "").strip().casefold()
        for i in range(self.pattern_list.count()):
            item = self.pattern_list.item(i)
            pattern = str(item.data(QtCore.Qt.UserRole) or "")
            item.setHidden(bool(query and query not in pattern.casefold()))

    def clear_pattern_filter(self):
        self.pattern_list.blockSignals(True)
        for i in range(self.pattern_list.count()):
            self.pattern_list.item(i).setCheckState(QtCore.Qt.Unchecked)
        self.pattern_list.blockSignals(False)
        self.apply_filter()

    def refresh_cases(self, select_first: bool = False):
        preserve = set(self._selected_patterns())
        current_path = self.current_case_path
        self.entries = scan_case_entries(self.case_root)
        self._rebuild_pattern_list(preserve)
        self.apply_filter(preferred_path=current_path, select_first=select_first)

    def apply_filter(self, *_args, preferred_path: Path | None = None, select_first: bool = False):
        if self._updating_case_list:
            return
        selected = self._selected_patterns()
        mode = str(self.mode_combo.currentData() or "ANY")
        self.filtered_entries = filter_case_entries(self.entries, selected, mode)

        target_path = preferred_path or self.current_case_path
        self._updating_case_list = True
        self.case_list.blockSignals(True)
        self.case_list.clear()
        target_row = -1
        for row, entry in enumerate(self.filtered_entries):
            item = QtWidgets.QListWidgetItem(entry.display_name)
            item.setData(QtCore.Qt.UserRole, str(entry.path))
            item.setToolTip(str(entry.path))
            # Keep unselected text high-contrast regardless of host OS palette.
            item.setForeground(QtGui.QBrush(QtGui.QColor("#f3f6fb")))
            self.case_list.addItem(item)
            if target_path is not None and entry.path == Path(target_path).resolve():
                target_row = row
        self.case_count_label.setText(
            f"Cases: {len(self.filtered_entries)} / {len(self.entries)}"
            + (f"   |   Filter: {len(selected)}" if selected else "")
        )
        self.case_list.blockSignals(False)
        self._updating_case_list = False

        if target_row >= 0:
            self.case_list.setCurrentRow(target_row)
        elif select_first and self.case_list.count() > 0:
            self.case_list.setCurrentRow(0)
        elif self.case_list.count() == 0:
            self._clear_current_case_display()

    def _case_item_changed(self, current, previous):
        if self._updating_case_list or current is None:
            return
        path = current.data(QtCore.Qt.UserRole)
        if path:
            self.open_case(Path(str(path)))

    def open_case(self, path: Path):
        try:
            case = self.repo.load(path)
            raw = self.market.load_for_case(path, case.market_data)
            if raw.empty:
                raise ValueError("Market data has no rows")
            replay = ReplayEngine.from_case(case, float(raw["timestamp"].max()))
            # Pattern Viewer always reveals the complete Case research window A -> C.
            replay.current_ts = min(float(replay.default_end_ts), float(replay.max_data_ts))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open Case Error", f"{path.name}\n\n{exc}")
            return

        self.current_case = case
        self.current_case_path = path.resolve()
        self.current_raw = raw
        self.current_replay = replay

        order_events, order_segments = build_order_overlay(case.orders, replay.current_ts)
        self.chart.set_order_events(order_events, order_segments, render=False)
        self.chart.set_context(case, raw, replay, str(path))
        self._update_rth_availability()
        self.chart.set_previous_rth_visible(self.show_rth_checkbox.isChecked())
        self.chart.set_all_intraday_notes_visible(self.show_notes_checkbox.isChecked())
        self.chart.set_all_orders_visible(self.show_orders_checkbox.isChecked())
        self.chart.render(reset_x=True)

        patterns = [str(x.get("text", "")).strip() for x in case.patterns if isinstance(x, dict) and str(x.get("text", "")).strip()]
        self.selected_path_label.setText(path.name)
        self.selected_path_label.setToolTip(str(path))
        self.selected_patterns_label.setText("Pattern: " + (" / ".join(patterns) if patterns else "無"))
        self._refresh_case_pattern_editor()
        self.case_status.setText(f"{case.case.get('symbol', '')}   {case.case.get('research_date', '')}")
        self.range_status.setText(
            f"Full: {case.time_range.get('data_start', '')}  →  {case.time_range.get('default_end', '')}"
        )
        self.setWindowTitle(f"Pattern Viewer v2.8 — {case.case.get('symbol', '')} — {path.name}")

    def _update_rth_availability(self):
        case = self.current_case
        rth = None if case is None else getattr(case, "reference_levels", {}).get("previous_rth")
        available = isinstance(rth, dict) and all(rth.get(k) is not None for k in ("high", "low", "close"))
        self.show_rth_checkbox.setEnabled(bool(available))
        if not available:
            self.show_rth_checkbox.blockSignals(True)
            self.show_rth_checkbox.setChecked(False)
            self.show_rth_checkbox.blockSignals(False)

    def _update_note_visibility(self, checked: bool):
        if self.current_case is not None:
            self.chart.set_all_intraday_notes_visible(bool(checked))

    def _update_rth_visibility(self, checked: bool):
        if self.current_case is not None:
            self.chart.set_previous_rth_visible(bool(checked))

    def _update_order_visibility(self, checked: bool):
        if self.current_case is not None:
            self.chart.set_all_orders_visible(bool(checked))

    def _clear_current_case_display(self):
        self.current_case = None
        self.current_case_path = None
        self.current_replay = None
        self.current_raw = pd.DataFrame()
        self.selected_path_label.setText("--")
        self.selected_patterns_label.setText("Pattern: --")
        self._refresh_case_pattern_editor()
        self.case_status.setText("沒有符合條件的 Case")
        self.range_status.setText("")
        self.setWindowTitle("Pattern Viewer v2.8")
