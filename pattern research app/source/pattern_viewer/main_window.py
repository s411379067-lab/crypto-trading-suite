from __future__ import annotations

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
    FilterCondition,
    ViewerCaseEntry,
    collect_patterns,
    filter_case_entries,
    save_case_patterns,
    scan_case_entries,
)
from .metrics import calculate_viewer_metrics
from .read_only_chart import ReadOnlyChartWidget


class PatternFilterButton(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.patterns: list[str] = []
        self.selected: set[str] = set()
        self._query = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.selection_button = QtWidgets.QPushButton("+ 選擇 Pattern")
        self.selection_button.setMinimumHeight(32)
        self.selection_button.clicked.connect(self._show_menu)
        layout.addWidget(self.selection_button)
        self.selected_rows_host = QtWidgets.QWidget(self)
        self.selected_rows_layout = QtWidgets.QVBoxLayout(self.selected_rows_host)
        self.selected_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.selected_rows_layout.setSpacing(2)
        layout.addWidget(self.selected_rows_host)
        self.selected_rows_host.hide()

    def set_patterns(self, patterns: list[str]):
        self.patterns = list(patterns)
        self.selected.intersection_update(self.patterns)
        self._update_text()

    def set_search_query(self, query: str):
        self._query = str(query or "").strip().casefold()

    def _update_text(self):
        selected = sorted(self.selected, key=str.casefold)
        while self.selected_rows_layout.count():
            item = self.selected_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for pattern in selected:
            row = QtWidgets.QWidget(self.selected_rows_host)
            row.setMinimumHeight(30)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(6, 0, 4, 0)
            row_layout.setSpacing(4)
            label = QtWidgets.QLabel(pattern)
            label.setMinimumHeight(28)
            remove = QtWidgets.QPushButton("×")
            remove.setFixedSize(24, 24)
            remove.clicked.connect(lambda _checked=False, item=pattern: self._set_selected(item, False))
            row_layout.addWidget(label, 1)
            row_layout.addWidget(remove)
            row.setStyleSheet("background:#172233; border:1px solid #394a64; border-radius:3px;")
            self.selected_rows_layout.addWidget(row)
        self.selected_rows_host.setVisible(bool(selected))
        self.setToolTip("\n".join(selected) if selected else "尚未選擇 Pattern")
        self.updateGeometry()

    def _show_menu(self):
        menu = QtWidgets.QMenu(self)
        matching = [p for p in self.patterns if not self._query or self._query in p.casefold()]
        if not matching:
            empty = menu.addAction("沒有符合的 Pattern")
            empty.setEnabled(False)
        for pattern in matching:
            action = menu.addAction(pattern)
            action.setCheckable(True)
            action.setChecked(pattern in self.selected)
            action.toggled.connect(lambda checked, item=pattern: self._set_selected(item, checked))
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def _set_selected(self, pattern: str, checked: bool):
        was_selected = pattern in self.selected
        if checked:
            self.selected.add(pattern)
        else:
            self.selected.discard(pattern)
        if was_selected != checked:
            self._update_text()
            self.changed.emit()


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
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        outer.addWidget(splitter, 1)

        self.sidebar = self._build_sidebar()
        self.chart = ReadOnlyChartWidget()
        self.chart.setMinimumWidth(650)
        self.inspector = self._build_inspector()
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.chart)
        splitter.addWidget(self.inspector)
        splitter.setSizes([360, 930, 310])
        splitter.setStretchFactor(1, 1)

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

        self.sidebar_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.sidebar_splitter.setChildrenCollapsible(False)
        self.sidebar_splitter.setHandleWidth(7)
        layout.addWidget(self.sidebar_splitter, 1)

        filter_box = QtWidgets.QGroupBox("Filters")
        fl = QtWidgets.QVBoxLayout(filter_box)
        fl.setSpacing(5)

        self.pattern_search = QtWidgets.QLineEdit()
        self.pattern_search.setPlaceholderText("搜尋 Pattern 名稱…")
        self.pattern_search.setMinimumHeight(32)
        fl.addWidget(self.pattern_search)

        combine_row = QtWidgets.QHBoxLayout()
        combine_row.addWidget(QtWidgets.QLabel("條件關係"))
        self.combine_combo = QtWidgets.QComboBox()
        self.combine_combo.addItem("AND — 全部符合", "AND")
        self.combine_combo.addItem("OR — 任一符合", "OR")
        self.combine_combo.setMinimumHeight(32)
        combine_row.addWidget(self.combine_combo, 1)
        self.btn_clear_patterns = QtWidgets.QPushButton("清除條件")
        combine_row.addWidget(self.btn_clear_patterns)
        fl.addLayout(combine_row)

        self.filter_rows_host = QtWidgets.QWidget()
        self.filter_rows_layout = QtWidgets.QVBoxLayout(self.filter_rows_host)
        self.filter_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_rows_layout.setSpacing(4)
        self.filter_rows_scroll = QtWidgets.QScrollArea()
        self.filter_rows_scroll.setWidgetResizable(True)
        self.filter_rows_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.filter_rows_scroll.setMinimumHeight(280)
        self.filter_rows_scroll.setMaximumHeight(340)
        self.filter_rows_scroll.setWidget(self.filter_rows_host)
        fl.addWidget(self.filter_rows_scroll)
        add_filter_row = QtWidgets.QHBoxLayout()
        self.btn_add_filter = QtWidgets.QPushButton("+ Filter")
        self.btn_add_filter.setToolTip("新增 Pattern、Symbol、日期或有無 Pattern 條件")
        add_filter_row.addWidget(self.btn_add_filter)
        add_filter_row.addStretch(1)
        fl.addLayout(add_filter_row)
        self.filter_rows: list[dict] = []

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
        filter_area = QtWidgets.QWidget()
        filter_area_layout = QtWidgets.QVBoxLayout(filter_area)
        filter_area_layout.setContentsMargins(0, 0, 0, 0)
        filter_area_layout.setSpacing(5)
        filter_area_layout.addWidget(filter_box, 1)
        filter_area_layout.addWidget(view_box)
        self.sidebar_splitter.addWidget(filter_area)

        self.case_count_label = QtWidgets.QLabel("Cases: 0")
        self.case_count_label.setStyleSheet("font-weight:700; color:#cfd7e6;")

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

        self.case_content_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.case_content_splitter.setChildrenCollapsible(False)
        self.case_content_splitter.setHandleWidth(7)
        self.case_content_splitter.addWidget(self.case_list)
        self.case_content_splitter.addWidget(self.pattern_edit_box)
        self.case_content_splitter.setSizes([650, 210])
        case_area = QtWidgets.QWidget()
        case_area_layout = QtWidgets.QVBoxLayout(case_area)
        case_area_layout.setContentsMargins(0, 0, 0, 0)
        case_area_layout.setSpacing(4)
        case_area_layout.addWidget(self.case_count_label)
        case_area_layout.addWidget(self.case_content_splitter, 1)
        self.sidebar_splitter.addWidget(case_area)
        self.sidebar_splitter.setSizes([500, 500])

        self.btn_folder.clicked.connect(self.choose_folder)
        self.btn_refresh.clicked.connect(lambda: self.refresh_cases(select_first=False))
        self.combine_combo.currentIndexChanged.connect(self.apply_filter)
        self.btn_add_filter.clicked.connect(lambda: self._add_filter_row())
        self.btn_clear_patterns.clicked.connect(self.clear_pattern_filter)
        self.pattern_search.textChanged.connect(self._apply_pattern_search_visibility)
        self.case_list.currentItemChanged.connect(self._case_item_changed)
        self.show_notes_checkbox.toggled.connect(self._update_note_visibility)
        self.show_rth_checkbox.toggled.connect(self._update_rth_visibility)
        self.show_orders_checkbox.toggled.connect(self._update_order_visibility)
        self.case_pattern_list.itemSelectionChanged.connect(self._case_pattern_selection_changed)
        self.case_pattern_list.itemDoubleClicked.connect(lambda _item: self.case_pattern_input.setFocus())
        self.btn_add_case_pattern.clicked.connect(self.add_case_pattern)
        self.btn_rename_case_pattern.clicked.connect(self.rename_case_pattern)
        self.btn_delete_case_pattern.clicked.connect(self.delete_case_pattern)
        self._add_filter_row("PATTERN")
        return panel

    def _build_inspector(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(260)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(7)
        self.inspector_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.inspector_splitter.setChildrenCollapsible(False)
        self.inspector_splitter.setHandleWidth(7)

        metrics_box = QtWidgets.QGroupBox("Filtered Case Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_box)
        metrics_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        metrics_layout.setFormAlignment(QtCore.Qt.AlignTop)
        self.metric_labels: dict[str, QtWidgets.QLabel] = {}
        for key, title in (
            ("profit_factor", "PF"),
            ("win_rate", "Win Rate"),
            ("pnl_per_day", "PnL / Day"),
            ("pnl_per_trade", "PnL / Trade"),
        ):
            value = QtWidgets.QLabel("--")
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            value.setStyleSheet("font-weight:700; color:#dbe9ff;")
            self.metric_labels[key] = value
            metrics_layout.addRow(title, value)
        self.metric_summary_label = QtWidgets.QLabel("0 trades · 0 days")
        self.metric_summary_label.setStyleSheet("color:#aebbd0;")
        metrics_layout.addRow("Sample", self.metric_summary_label)
        self.inspector_splitter.addWidget(metrics_box)

        detail_box = QtWidgets.QGroupBox("Selected Case")
        detail_layout = QtWidgets.QVBoxLayout(detail_box)
        self.selected_path_label = QtWidgets.QLabel("--")
        self.selected_path_label.setWordWrap(True)
        self.selected_patterns_label = QtWidgets.QLabel("Pattern: --")
        self.selected_patterns_label.setWordWrap(True)
        self.selected_pnl_label = QtWidgets.QLabel("Realized PnL: --")
        self.selected_pnl_label.setStyleSheet("color:#cfd7e6; font-weight:700;")
        detail_layout.addWidget(self.selected_path_label)
        detail_layout.addWidget(self.selected_patterns_label)
        detail_layout.addWidget(self.selected_pnl_label)
        self.inspector_splitter.addWidget(detail_box)
        self.inspector_splitter.setSizes([220, 500])
        layout.addWidget(self.inspector_splitter, 1)
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

    def _add_filter_row(self, field: str = "SYMBOL"):
        widget = QtWidgets.QWidget(self.filter_rows_host)
        row_layout = QtWidgets.QVBoxLayout(widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        field_combo = QtWidgets.QComboBox()
        field_combo.setMinimumHeight(32)
        for label, key in (
            ("Pattern", "PATTERN"),
            ("Symbol", "SYMBOL"),
            ("日期", "DATE"),
            ("有 Pattern", "HAS_PATTERN"),
        ):
            field_combo.addItem(label, key)
        field_index = field_combo.findData(field)
        field_combo.setCurrentIndex(max(0, field_index))
        operator_combo = QtWidgets.QComboBox()
        operator_combo.setMinimumHeight(32)
        value_slot = QtWidgets.QWidget()
        value_layout = QtWidgets.QHBoxLayout(value_slot)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.setSpacing(3)
        remove_button = QtWidgets.QPushButton("×")
        remove_button.setFixedSize(32, 32)
        header_layout.addWidget(field_combo)
        header_layout.addWidget(operator_combo, 1)
        header_layout.addWidget(remove_button)
        row_layout.addLayout(header_layout)
        row_layout.addWidget(value_slot)
        row = {
            "widget": widget,
            "field": field_combo,
            "operator": operator_combo,
            "value_layout": value_layout,
            "value_slot": value_slot,
            "value": None,
        }
        self.filter_rows.append(row)
        self.filter_rows_layout.addWidget(widget)
        self._configure_filter_row(row)
        field_combo.currentIndexChanged.connect(lambda _index, item=row: self._on_filter_field_changed(item))
        operator_combo.currentIndexChanged.connect(lambda _index, item=row: self._on_filter_operator_changed(item))
        remove_button.clicked.connect(lambda _checked=False, item=row: self._remove_filter_row(item))
        return row

    def _configure_filter_row(self, row: dict):
        field = str(row["field"].currentData() or "PATTERN")
        row["operator"].blockSignals(True)
        row["operator"].clear()
        value_layout = row["value_layout"]
        while value_layout.count():
            item = value_layout.takeAt(0)
            old_widget = item.widget()
            if old_widget is not None:
                old_widget.deleteLater()

        value = None
        row["date_start"] = None
        row["date_end"] = None
        if field == "PATTERN":
            row["operator"].addItem("任一", "ANY")
            row["operator"].addItem("全部", "ALL")
            row["operator"].addItem("不包含任一", "EXCLUDE_ANY")
            row["operator"].addItem("不包含全部", "EXCLUDE_ALL")
            value = PatternFilterButton(row["value_slot"])
            value.set_patterns(collect_patterns(self.entries))
            value.set_search_query(self.pattern_search.text())
            value.changed.connect(self.apply_filter)
            value_layout.addWidget(value)
        elif field == "SYMBOL":
            row["operator"].addItem("包含", "CONTAINS")
            row["operator"].addItem("等於", "EQUALS")
            row["operator"].addItem("不包含", "NOT_CONTAINS")
            value = QtWidgets.QLineEdit(row["value_slot"])
            value.setPlaceholderText("輸入 Symbol")
            value.setMinimumHeight(32)
            value.textChanged.connect(self.apply_filter)
            value_layout.addWidget(value)
        elif field == "DATE":
            row["operator"].addItem("範圍內", "BETWEEN")
            row["operator"].addItem("截至", "BEFORE")
            row["operator"].addItem("自…起", "AFTER")
            value = QtWidgets.QWidget(row["value_slot"])
            date_layout = QtWidgets.QHBoxLayout(value)
            date_layout.setContentsMargins(0, 0, 0, 0)
            date_layout.setSpacing(3)
            start = QtWidgets.QDateEdit(value)
            start.setCalendarPopup(True)
            start.setDisplayFormat("yyyy-MM-dd")
            start.setDate(QtCore.QDate(1900, 1, 1))
            start.setMinimumHeight(30)
            end = QtWidgets.QDateEdit(value)
            end.setCalendarPopup(True)
            end.setDisplayFormat("yyyy-MM-dd")
            end.setDate(QtCore.QDate(9999, 12, 31))
            end.setMinimumHeight(30)
            date_layout.addWidget(start)
            date_layout.addWidget(end)
            start.dateChanged.connect(self.apply_filter)
            end.dateChanged.connect(self.apply_filter)
            row["date_start"] = start
            row["date_end"] = end
            value_layout.addWidget(value)
        else:
            row["operator"].addItem("不限", "NONE")
            row["operator"].addItem("有 Pattern", "HAS")
            row["operator"].addItem("沒有 Pattern", "HAS_NOT")

        row["value"] = value
        row["operator"].blockSignals(False)
        self._update_date_filter_visibility(row)

    def _update_date_filter_visibility(self, row: dict):
        if row["field"].currentData() != "DATE" or row["value"] is None:
            return
        start = row["date_start"]
        end = row["date_end"]
        operator = row["operator"].currentData()
        start.setVisible(operator != "BEFORE")
        end.setVisible(operator != "AFTER")

    def _on_filter_field_changed(self, row: dict):
        self._configure_filter_row(row)
        self.apply_filter()

    def _on_filter_operator_changed(self, row: dict):
        self._update_date_filter_visibility(row)
        self.apply_filter()

    def _remove_filter_row(self, row: dict):
        if row not in self.filter_rows:
            return
        self.filter_rows.remove(row)
        row["widget"].deleteLater()
        self.apply_filter()

    def _filter_conditions(self) -> list[FilterCondition]:
        conditions = []
        for row in self.filter_rows:
            field = str(row["field"].currentData() or "")
            operator = str(row["operator"].currentData() or "")
            value = row["value"]
            if field == "PATTERN":
                conditions.append(FilterCondition(field, operator, tuple(sorted(value.selected))))
            elif field == "SYMBOL":
                conditions.append(FilterCondition(field, operator, value=value.text()))
            elif field == "DATE":
                conditions.append(FilterCondition(
                    field,
                    operator,
                    start=row["date_start"].date().toString("yyyy-MM-dd"),
                    end=row["date_end"].date().toString("yyyy-MM-dd"),
                ))
            else:
                conditions.append(FilterCondition(field, operator))
        return conditions

    def _rebuild_pattern_list(self):
        patterns = collect_patterns(self.entries)
        for row in self.filter_rows:
            value = row.get("value")
            if isinstance(value, PatternFilterButton):
                value.set_patterns(patterns)
        self._apply_pattern_search_visibility(self.pattern_search.text())

    def _apply_pattern_search_visibility(self, text: str):
        query = str(text or "").strip().casefold()
        for row in self.filter_rows:
            value = row.get("value")
            if isinstance(value, PatternFilterButton):
                value.set_search_query(query)

    def clear_pattern_filter(self):
        for row in list(self.filter_rows):
            self._remove_filter_row(row)
        self._add_filter_row("PATTERN")
        self.apply_filter()

    def refresh_cases(self, select_first: bool = False):
        current_path = self.current_case_path
        self.entries = scan_case_entries(self.case_root)
        self._rebuild_pattern_list()
        self.apply_filter(preferred_path=current_path, select_first=select_first)

    def apply_filter(self, *_args, preferred_path: Path | None = None, select_first: bool = False):
        if self._updating_case_list:
            return
        conditions = self._filter_conditions()
        combine = str(self.combine_combo.currentData() or "AND")
        self.filtered_entries = filter_case_entries(self.entries, conditions, combine)
        self._update_filtered_metrics()

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
            + (f"   |   Filters: {len(self.filter_rows)} {combine}" if self.filter_rows else "")
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
        realized_pnl = sum(float(segment["pnl"]) for segment in order_segments)
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
        self._set_selected_realized_pnl(realized_pnl)
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

    def _set_selected_realized_pnl(self, realized_pnl: float):
        if realized_pnl > 1e-12:
            color, value = "#7bd88f", f"+{realized_pnl:.3f}"
        elif realized_pnl < -1e-12:
            color, value = "#ff6b6b", f"{realized_pnl:.3f}"
        else:
            color, value = "#cfd7e6", "0.000"
        self.selected_pnl_label.setText(f"Realized PnL: {value}")
        self.selected_pnl_label.setStyleSheet(f"color:{color}; font-weight:700;")

    def _clear_current_case_display(self):
        self.current_case = None
        self.current_case_path = None
        self.current_replay = None
        self.current_raw = pd.DataFrame()
        self.selected_path_label.setText("--")
        self.selected_patterns_label.setText("Pattern: --")
        self.selected_pnl_label.setText("Realized PnL: --")
        self.selected_pnl_label.setStyleSheet("color:#cfd7e6; font-weight:700;")
        self._refresh_case_pattern_editor()
        self.case_status.setText("沒有符合條件的 Case")
        self.range_status.setText("")
        self.setWindowTitle("Pattern Viewer v2.8")

    def _update_filtered_metrics(self):
        metrics = calculate_viewer_metrics(
            (entry.research_date, entry.trade_pnls) for entry in self.filtered_entries
        )
        self.metric_labels["profit_factor"].setText(
            "∞" if metrics.profit_factor == float("inf") else self._format_metric(metrics.profit_factor)
        )
        self.metric_labels["win_rate"].setText(
            "--" if metrics.win_rate is None else f"{metrics.win_rate * 100:.1f}% ({metrics.wins}/{metrics.trades})"
        )
        self.metric_labels["pnl_per_day"].setText(self._format_metric(metrics.pnl_per_day))
        self.metric_labels["pnl_per_trade"].setText(self._format_metric(metrics.pnl_per_trade))
        self.metric_summary_label.setText(f"{metrics.trades} trades · {metrics.trading_days} days")

    @staticmethod
    def _format_metric(value: float | None) -> str:
        return "--" if value is None else f"{value:,.2f}"
