from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.repository import CaseRepository
from shared_core.market_data import MarketDataService
from shared_core.replay import ReplayEngine, ts_to_iso
from shared_core.undo import ResearchUndoManager
from .case_library import CaseLibraryWidget
from .chart_widget import ChartWidget
from .research_panel import ResearchPanel
from .order_panel import OrderPanel


APP_STYLE = """
QWidget { background-color:#0f131c; color:#d7deea; }
QPushButton { background-color:#192131; border:1px solid #4d5a73; border-radius:3px; padding:5px 8px; font-weight:600; }
QPushButton:hover { background-color:#253249; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QListWidget, QTreeView {
    background-color:#0d1420; border:1px solid #36445d; border-radius:3px; color:#e6edf7; padding:4px;
}
QTableWidget {
    background-color:#0d1420;
    alternate-background-color:#111a29;
    color:#e6edf7;
    gridline-color:#2d3748;
    border:1px solid #36445d;
    selection-background-color:#2a3952;
    selection-color:#ffffff;
    font-size:10pt;
}
QHeaderView::section {
    background-color:#263247;
    color:#ffffff;
    border:0px;
    border-right:1px solid #4d5a73;
    border-bottom:1px solid #4d5a73;
    padding:6px 5px;
    font-size:10pt;
    font-weight:700;
}
QHeaderView::section:hover { background-color:#31405a; }
QTableCornerButton::section {
    background-color:#263247;
    border:1px solid #4d5a73;
}
QGroupBox { border:1px solid #2a3142; border-radius:4px; margin-top:8px; padding-top:8px; font-weight:700; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }
QSplitter::handle { background:#252c3b; }
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, case_root: str | Path):
        super().__init__()
        self.setWindowTitle("Pattern Analyzer v2.8")
        self.resize(1550, 900)
        self.setStyleSheet(APP_STYLE)

        self.repo = CaseRepository()
        self.market = MarketDataService()
        self.case = None
        self.case_path: Path | None = None
        self.raw_df = pd.DataFrame()
        self.replay = None
        self.dirty = False
        self.undo_manager = ResearchUndoManager(max_depth=200)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left side is intentionally split into two independently resizable blocks:
        # Order on top, Case Library on bottom.
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.order = OrderPanel()
        self.library = CaseLibraryWidget(case_root)
        self.left_splitter.addWidget(self.order)
        self.left_splitter.addWidget(self.library)
        self.left_splitter.setSizes([560, 300])

        self.chart = ChartWidget()
        self.research = ResearchPanel()
        self.splitter.addWidget(self.left_splitter)
        self.splitter.addWidget(self.chart)
        self.splitter.addWidget(self.research)
        self.splitter.setSizes([390, 900, 330])
        layout.addWidget(self.splitter, 1)

        replay_bar = QtWidgets.QWidget()
        rb = QtWidgets.QHBoxLayout(replay_bar)
        rb.setContentsMargins(4, 2, 4, 2)
        self.btn_back = QtWidgets.QPushButton("◀")
        self.btn_forward = QtWidgets.QPushButton("▶")
        self.btn_reset = QtWidgets.QPushButton("Reset B")
        self.replay_label = QtWidgets.QLabel("Replay: --")
        self.range_label = QtWidgets.QLabel("A -- / B -- / C --")
        self.save_label = QtWidgets.QLabel("Saved")
        self.save_label.setStyleSheet("color:#7bd88f;")
        rb.addWidget(self.btn_back)
        rb.addWidget(self.btn_forward)
        rb.addWidget(self.btn_reset)
        rb.addWidget(self.replay_label)
        rb.addSpacing(20)
        rb.addWidget(self.range_label)
        rb.addStretch(1)
        rb.addWidget(self.save_label)
        layout.addWidget(replay_bar)

        self.library.case_open_requested.connect(self.open_case)
        self.chart.dirty.connect(self.mark_dirty)
        self.chart.history_committed.connect(self._record_history)
        self.research.changed.connect(self.mark_dirty)
        self.research.history_committed.connect(self._record_history)
        self.research.reference_levels_changed.connect(self.chart.set_previous_rth_visible)
        self.research.note_selection_changed.connect(self.chart.set_selected_intraday_note)
        self.research.all_notes_visibility_changed.connect(self.chart.set_all_intraday_notes_visible)
        self.order.changed.connect(self.mark_dirty)
        self.order.fills_changed.connect(self._refresh_order_markers)
        self.order.pending_bracket_changed.connect(self.chart.set_pending_bracket)
        self.chart.pending_bracket_edited.connect(self.order.update_pending_bracket)
        self.chart.pending_bracket_submit_requested.connect(self.order.submit_pending_bracket)
        self.chart.pending_bracket_cancel_requested.connect(self.order.cancel_pending_bracket)
        self.order.active_position_changed.connect(self.chart.set_active_position)
        self.order.submitted_bracket_changed.connect(self.chart.set_submitted_bracket)
        self.chart.submitted_bracket_draft_changed.connect(self.order.update_submitted_bracket_draft)
        self.order.submitted_bracket_draft_changed.connect(self.chart.set_submitted_bracket_draft)
        self.chart.submitted_bracket_apply_requested.connect(self.order.apply_submitted_bracket_draft)
        self.chart.submitted_bracket_revert_requested.connect(self.order.cancel_submitted_bracket_draft)
        self.chart.active_position_close_requested.connect(self.order.close_position)
        self.chart.submitted_bracket_cancel_requested.connect(self.order.cancel_submitted_bracket)
        self.chart.submitted_group_cancel_requested.connect(self.order.cancel_submitted_group)
        self.chart.order_prefill_requested.connect(self.order.prefill_from_chart)
        self.btn_forward.clicked.connect(self.step_forward)
        self.btn_back.clicked.connect(self.step_backward)
        self.btn_reset.clicked.connect(self.reset_replay)

        self.shortcut_forward = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        self.shortcut_forward.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_forward.activated.connect(self.step_forward)
        self.shortcut_back = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        self.shortcut_back.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_back.activated.connect(self.step_backward)
        self.shortcut_delete = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        self.shortcut_delete.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_delete.activated.connect(self.chart.delete_selected_drawing)

        self.shortcut_undo = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_undo.activated.connect(self.undo)
        self.shortcut_redo = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_redo.activated.connect(self.redo)

        self.shortcut_clear_note = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        self.shortcut_clear_note.setContext(QtCore.Qt.WindowShortcut)
        self.shortcut_clear_note.activated.connect(self.research.clear_note_selection)

        self.autosave = QtCore.QTimer(self)
        self.autosave.setInterval(1000)
        self.autosave.timeout.connect(self.save_if_dirty)
        self.autosave.start()

    def open_case(self, path: str):
        self.save_if_dirty(force=True)
        try:
            case = self.repo.load(path)
            raw = self.market.load_for_case(path, case.market_data)
            if raw.empty:
                raise ValueError("Market data has no rows")
            replay = ReplayEngine.from_case(case, float(raw["timestamp"].max()))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Case Error", str(e))
            return

        self.case = case
        self.case_path = Path(path)
        self.raw_df = raw
        self.replay = replay
        self.dirty = False
        self.order.set_context(case, raw, replay)
        self.chart.set_order_events(self.order.visible_fill_events(), self.order.visible_trade_segments(), render=False)
        self.chart.set_context(case, raw, replay, path)
        self.research.set_case(case, self.current_replay_time_text)
        self.chart.set_previous_rth_visible(self.research.rth_checkbox.isChecked())
        # Rendering can normalize legacy Drawing fields; start history from the normalized state.
        self.undo_manager.reset(case)
        self.update_status()
        self.setWindowTitle(f"Pattern Analyzer v2.8 — {case.case.get('symbol')} — {self.case_path.name}")

    def _record_history(self, label: str):
        if self.case is None:
            return
        self.undo_manager.record(self.case, label)

    @staticmethod
    def _focused_text_editor():
        widget = QtWidgets.QApplication.focusWidget()
        if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            return widget
        return None

    def undo(self):
        editor = self._focused_text_editor()
        if editor is not None:
            try:
                editor.undo()
            except Exception:
                pass
            return
        if self.case is None:
            return
        label = self.undo_manager.undo(self.case)
        if label is None:
            return
        self.case.touch()
        self._refresh_after_history_restore()
        self.mark_dirty()

    def redo(self):
        editor = self._focused_text_editor()
        if editor is not None:
            try:
                editor.redo()
            except Exception:
                pass
            return
        if self.case is None:
            return
        label = self.undo_manager.redo(self.case)
        if label is None:
            return
        self.case.touch()
        self._refresh_after_history_restore()
        self.mark_dirty()

    def _refresh_after_history_restore(self):
        if self.case is None:
            return
        valid_ids = {str(d.get("id")) for d in self.case.drawings}
        if self.chart.selected_drawing_id not in valid_ids:
            self.chart.selected_drawing_id = None
        self.chart.render(reset_x=False)
        self.research.refresh()

    def current_replay_time_text(self) -> str:
        if self.replay is None or self.case is None:
            return ""
        tz = self.case.display.get("timezone", "Asia/Taipei")
        dt = pd.Timestamp(self.replay.current_ts, unit="s", tz="UTC").tz_convert(tz)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    def step_forward(self):
        if self.replay is None:
            return
        previous_ts = float(self.replay.current_ts)
        self.replay.forward()
        self.order.process_replay_advance(previous_ts, float(self.replay.current_ts))
        self._sync_replay_to_case()
        self.chart.set_order_events(self.order.visible_fill_events(), self.order.visible_trade_segments(), render=False)
        self.chart.render(reset_x=False)
        self.order.refresh()
        self.mark_dirty()
        self.update_status()

    def step_backward(self):
        if self.replay is None:
            return
        self.replay.backward()
        self._sync_replay_to_case()
        self.order.refresh()
        self.chart.set_order_events(self.order.visible_fill_events(), self.order.visible_trade_segments(), render=False)
        self.chart.render(reset_x=False)
        self.mark_dirty()
        self.update_status()

    def reset_replay(self):
        if self.replay is None:
            return
        self.replay.reset()
        self._sync_replay_to_case()
        self.order.refresh()
        self.chart.set_order_events(self.order.visible_fill_events(), self.order.visible_trade_segments(), render=False)
        self.chart.render(reset_x=True)
        self.mark_dirty()
        self.update_status()

    def _refresh_order_markers(self):
        if self.case is None:
            return
        self.chart.set_order_events(self.order.visible_fill_events(), self.order.visible_trade_segments(), render=False)
        self.chart.render(reset_x=False)

    def _sync_replay_to_case(self):
        if self.case is None or self.replay is None:
            return
        self.case.replay["current_time"] = ts_to_iso(self.replay.current_ts)
        self.case.touch()

    def mark_dirty(self):
        self.dirty = True
        self.save_label.setText("Unsaved")
        self.save_label.setStyleSheet("color:#ffcc80;")

    def save_if_dirty(self, force=False):
        if self.case is None or self.case_path is None:
            return
        if not self.dirty and not force:
            return
        try:
            self.chart.sync_all_drawings_from_view()
            self._sync_replay_to_case()
            self.repo.save(self.case_path, self.case)
            self.dirty = False
            self.save_label.setText("Saved")
            self.save_label.setStyleSheet("color:#7bd88f;")
        except Exception as e:
            self.save_label.setText("Save Error")
            self.save_label.setStyleSheet("color:#f7525f;")
            if force:
                QtWidgets.QMessageBox.warning(self, "Save Error", str(e))

    def update_status(self):
        if self.case is None or self.replay is None:
            return
        self.replay_label.setText(f"Replay: {self.current_replay_time_text()}")
        tr = self.case.time_range
        self.range_label.setText(f"A {tr['data_start']}   /   B {tr['replay_start']}   /   C {tr['default_end']}")

    def closeEvent(self, event):
        self.save_if_dirty(force=True)
        super().closeEvent(event)
