from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from shared_core.models import utc_now_iso


class ResearchPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()
    reference_levels_changed = QtCore.Signal(bool)
    history_committed = QtCore.Signal(str)
    note_selection_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.case = None
        self.replay_time_provider = lambda: ""
        self._editing_note_id = None
        self._note_editor = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Pattern Research")
        title.setStyleSheet("font-size:13pt; font-weight:700; color:#e6edf7;")
        layout.addWidget(title)

        ref_box = QtWidgets.QGroupBox("Reference Levels")
        ref_layout = QtWidgets.QHBoxLayout(ref_box)
        self.rth_checkbox = QtWidgets.QCheckBox("Previous RTH H/L")
        self.rth_status = QtWidgets.QLabel("[無資料]")
        self.rth_status.setWordWrap(False)
        self.rth_status.setStyleSheet("color:#8f9bad; font-size:9.5pt;")
        ref_layout.addWidget(self.rth_checkbox)
        ref_layout.addWidget(self.rth_status, 1)
        layout.addWidget(ref_box)

        pattern_box = QtWidgets.QGroupBox("Pattern")
        p_layout = QtWidgets.QVBoxLayout(pattern_box)
        self.pattern_input = QtWidgets.QLineEdit()
        self.pattern_input.setPlaceholderText("例如：開盤TR BO 1R後反轉")
        self.btn_add_pattern = QtWidgets.QPushButton("新增 Pattern")
        self.pattern_list = QtWidgets.QListWidget()
        self.btn_delete_pattern = QtWidgets.QPushButton("刪除選取 Pattern")
        p_layout.addWidget(self.pattern_input)
        p_layout.addWidget(self.btn_add_pattern)
        p_layout.addWidget(self.pattern_list, 1)
        p_layout.addWidget(self.btn_delete_pattern)
        layout.addWidget(pattern_box, 1)

        notes_box = QtWidgets.QGroupBox("盤中紀錄")
        n_layout = QtWidgets.QVBoxLayout(notes_box)
        self.note_input = QtWidgets.QTextEdit()
        self.note_input.setPlaceholderText("記錄目前 Replay 時點看到的結構與想法…")
        self.note_input.setMaximumHeight(100)
        self.btn_add_note = QtWidgets.QPushButton("新增盤中紀錄")
        self.note_list = QtWidgets.QListWidget()
        self.btn_delete_note = QtWidgets.QPushButton("刪除選取紀錄")
        n_layout.addWidget(self.note_input)
        n_layout.addWidget(self.btn_add_note)
        n_layout.addWidget(self.note_list, 2)
        n_layout.addWidget(self.btn_delete_note)
        layout.addWidget(notes_box, 2)

        self.btn_add_pattern.clicked.connect(self.add_pattern)
        self.pattern_input.returnPressed.connect(self.add_pattern)
        self.btn_delete_pattern.clicked.connect(self.delete_pattern)
        self.btn_add_note.clicked.connect(self.add_note)
        self.btn_delete_note.clicked.connect(self.delete_note)
        self.note_list.itemDoubleClicked.connect(self._begin_note_edit)
        self.note_list.itemSelectionChanged.connect(self._emit_note_selection)
        self.rth_checkbox.toggled.connect(self._rth_toggled)

    def set_case(self, case, replay_time_provider):
        self.case = case
        self.replay_time_provider = replay_time_provider
        self.refresh(preserve_note_selection=False)

    def refresh(self, preserve_note_selection=True):
        selected_note_id = None
        if preserve_note_selection:
            current = self.note_list.currentItem()
            if current is not None:
                selected_note_id = current.data(QtCore.Qt.UserRole)
        self._editing_note_id = None
        self._note_editor = None
        self.pattern_list.clear()
        self.note_list.blockSignals(True)
        self.note_list.clear()
        if self.case is None:
            self.rth_checkbox.blockSignals(True)
            self.rth_checkbox.setChecked(False)
            self.rth_checkbox.setEnabled(False)
            self.rth_checkbox.blockSignals(False)
            self.rth_status.setText("[無資料]")
            self.note_list.blockSignals(False)
            self.note_selection_changed.emit(None)
            return

        for item in self.case.patterns:
            widget_item = QtWidgets.QListWidgetItem(item.get("text", ""))
            widget_item.setData(QtCore.Qt.UserRole, item.get("id"))
            self.pattern_list.addItem(widget_item)
        selected_row = -1
        for item in self.case.intraday_notes:
            stamp = item.get("replay_time", "")
            text = item.get("text", "")
            widget_item = QtWidgets.QListWidgetItem(f"{stamp}\n{text}")
            widget_item.setData(QtCore.Qt.UserRole, item.get("id"))
            self.note_list.addItem(widget_item)
            if selected_note_id is not None and item.get("id") == selected_note_id:
                selected_row = self.note_list.count() - 1
        if selected_row >= 0:
            self.note_list.setCurrentRow(selected_row)
        self.note_list.blockSignals(False)
        self._emit_note_selection()

        rth = self.case.reference_levels.get("previous_rth") if hasattr(self.case, "reference_levels") else None
        available = isinstance(rth, dict) and rth.get("high") is not None and rth.get("low") is not None and rth.get("close") is not None
        self.rth_checkbox.blockSignals(True)
        self.rth_checkbox.setEnabled(bool(available))
        self.rth_checkbox.setChecked(bool(available and self.case.display.get("show_previous_rth", False)))
        self.rth_checkbox.blockSignals(False)
        if available:
            self.rth_status.setText(
                f"{rth.get('session_date', '')}   H {float(rth['high']):.2f}   L {float(rth['low']):.2f}   C {float(rth['close']):.2f}"
            )
            self.rth_status.setStyleSheet("color:#cfd7e6; font-size:9.5pt;")
        else:
            is_weekend = bool(getattr(self.case, "calendar", {}).get("is_weekend", False))
            self.rth_status.setText("[週末 / 無資料]" if is_weekend else "[無資料]")
            self.rth_status.setStyleSheet("color:#8f9bad; font-size:9.5pt;")

    def _rth_toggled(self, checked: bool):
        if self.case is None:
            return
        self.case.display["show_previous_rth"] = bool(checked)
        self.case.touch()
        self.reference_levels_changed.emit(bool(checked))
        self.changed.emit()

    def add_pattern(self):
        if self.case is None:
            return
        text = self.pattern_input.text().strip()
        if not text:
            return
        self.case.add_pattern(text)
        self.pattern_input.clear()
        self.refresh()
        self.changed.emit()
        self.history_committed.emit("Add Pattern")

    def delete_pattern(self):
        if self.case is None:
            return
        item = self.pattern_list.currentItem()
        if item is None:
            return
        target_id = item.data(QtCore.Qt.UserRole)
        self.case.patterns = [x for x in self.case.patterns if x.get("id") != target_id]
        self.case.touch()
        self.refresh()
        self.changed.emit()
        self.history_committed.emit("Delete Pattern")

    def add_note(self):
        if self.case is None:
            return
        text = self.note_input.toPlainText().strip()
        if not text:
            return
        self.case.add_note(self.replay_time_provider(), text)
        self.note_input.clear()
        self.refresh()
        self.changed.emit()
        self.history_committed.emit("Add Note")

    def delete_note(self):
        if self.case is None:
            return
        item = self.note_list.currentItem()
        if item is None:
            return
        target_id = item.data(QtCore.Qt.UserRole)
        self.case.intraday_notes = [x for x in self.case.intraday_notes if x.get("id") != target_id]
        self.case.touch()
        self.refresh()
        self.changed.emit()
        self.history_committed.emit("Delete Note")

    def _emit_note_selection(self):
        if self.case is None:
            self.note_selection_changed.emit(None)
            return
        item = self.note_list.currentItem()
        if item is None:
            self.note_selection_changed.emit(None)
            return
        note = self._note_by_id(item.data(QtCore.Qt.UserRole))
        self.note_selection_changed.emit(dict(note) if note is not None else None)

    def _note_by_id(self, note_id):
        if self.case is None:
            return None
        for note in self.case.intraday_notes:
            if note.get("id") == note_id:
                return note
        return None

    def _begin_note_edit(self, item):
        """Edit a note in-place inside the existing list row.

        No permanent editor UI is added: the selected row is temporarily replaced
        by timestamp + text editor + Save/Cancel buttons.
        """
        if self.case is None or item is None:
            return
        note_id = item.data(QtCore.Qt.UserRole)
        note = self._note_by_id(note_id)
        if note is None:
            return
        if self._editing_note_id == note_id and self._note_editor is not None:
            self._note_editor.setFocus(QtCore.Qt.MouseFocusReason)
            return

        # Starting another edit cancels the previous unsaved inline edit.
        if self._editing_note_id is not None and self._editing_note_id != note_id:
            self.refresh()
            # refresh rebuilt the list, so locate the requested row again.
            for row in range(self.note_list.count()):
                candidate = self.note_list.item(row)
                if candidate.data(QtCore.Qt.UserRole) == note_id:
                    item = candidate
                    break

        self._editing_note_id = note_id
        editor_row = QtWidgets.QWidget(self.note_list)
        row_layout = QtWidgets.QHBoxLayout(editor_row)
        row_layout.setContentsMargins(4, 3, 4, 3)
        row_layout.setSpacing(6)

        stamp = QtWidgets.QLabel(str(note.get("replay_time", "")))
        stamp.setStyleSheet("color:#8f9bad; font-size:9pt;")
        stamp.setWordWrap(True)
        stamp.setMaximumWidth(145)

        editor = QtWidgets.QPlainTextEdit()
        editor.setPlainText(str(note.get("text", "")))
        editor.setMinimumHeight(64)
        editor.setTabChangesFocus(True)

        save_btn = QtWidgets.QPushButton("儲存")
        cancel_btn = QtWidgets.QPushButton("取消")
        save_btn.setFixedWidth(52)
        cancel_btn.setFixedWidth(52)

        row_layout.addWidget(stamp)
        row_layout.addWidget(editor, 1)
        row_layout.addWidget(save_btn)
        row_layout.addWidget(cancel_btn)
        self.note_list.setItemWidget(item, editor_row)
        item.setSizeHint(editor_row.sizeHint().expandedTo(QtCore.QSize(0, 74)))
        self._note_editor = editor

        save_btn.clicked.connect(lambda _=False, nid=note_id: self._save_note_edit(nid))
        cancel_btn.clicked.connect(self._cancel_note_edit)
        editor.setFocus(QtCore.Qt.MouseFocusReason)
        editor.selectAll()

    def _save_note_edit(self, note_id):
        if self.case is None or self._editing_note_id != note_id or self._note_editor is None:
            return
        note = self._note_by_id(note_id)
        if note is None:
            self.refresh()
            return
        text = self._note_editor.toPlainText().strip()
        if not text:
            return
        if text == str(note.get("text", "")):
            self.refresh()
            return
        note["text"] = text
        note["updated_at"] = utc_now_iso()
        self.case.touch()
        self.refresh()
        self.changed.emit()
        self.history_committed.emit("Edit Note")

    def _cancel_note_edit(self):
        self.refresh()
