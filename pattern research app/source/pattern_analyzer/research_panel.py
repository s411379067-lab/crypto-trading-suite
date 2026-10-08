from __future__ import annotations

import html

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.models import utc_now_iso


class DeselectableNoteListWidget(QtWidgets.QListWidget):
    """QListWidget where a second single-click on the selected row deselects it.

    A real double-click still behaves normally so the parent can open the
    inline note editor without leaving the row deselected.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clear_on_release_id = None
        self._press_pos = None

    @staticmethod
    def _event_pos(event):
        try:
            return event.position().toPoint()
        except Exception:
            return event.pos()

    def mousePressEvent(self, event):
        self._clear_on_release_id = None
        self._press_pos = self._event_pos(event)
        if event.button() == QtCore.Qt.LeftButton:
            item = self.itemAt(self._press_pos)
            if item is not None and item.isSelected():
                self._clear_on_release_id = item.data(QtCore.Qt.UserRole)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Do not let the second release of a double-click toggle selection off.
        self._clear_on_release_id = None
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        release_pos = self._event_pos(event)
        pending_id = self._clear_on_release_id
        press_pos = self._press_pos
        self._clear_on_release_id = None
        self._press_pos = None
        super().mouseReleaseEvent(event)

        if event.button() != QtCore.Qt.LeftButton or pending_id is None:
            return
        item = self.itemAt(release_pos)
        if item is None or item.data(QtCore.Qt.UserRole) != pending_id:
            return
        if press_pos is not None:
            delta = release_pos - press_pos
            if delta.manhattanLength() > QtWidgets.QApplication.startDragDistance():
                return
        self.clearSelection()
        try:
            self.selectionModel().clearCurrentIndex()
        except Exception:
            pass



class IntradayNoteDelegate(QtWidgets.QStyledItemDelegate):
    """Render each note as a compact, two-tone timestamp / content row."""

    TIME_ROLE = QtCore.Qt.UserRole + 1
    TEXT_ROLE = QtCore.Qt.UserRole + 2

    def _document(self, option, index):
        timestamp = html.escape(str(index.data(self.TIME_ROLE) or ""))
        text = html.escape(str(index.data(self.TEXT_ROLE) or "")).replace("\n", "<br>")
        document = QtGui.QTextDocument()
        document.setDefaultFont(option.font)
        document.setHtml(
            f'<span style="color:#62c9ff">{timestamp}</span>'
            f'<br><span style="color:#e6edf7">{text}</span>'
        )
        document.setTextWidth(max(20.0, float(option.rect.width()) - 12.0))
        return document

    def paint(self, painter, option, index):
        item_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        item_option.text = ""
        style = item_option.widget.style() if item_option.widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, item_option, painter, item_option.widget)

        document = self._document(option, index)
        painter.save()
        painter.setClipRect(option.rect)
        painter.translate(option.rect.left() + 6, option.rect.top() + 3)
        document.documentLayout().draw(painter, QtGui.QAbstractTextDocumentLayout.PaintContext())
        painter.restore()

    def sizeHint(self, option, index):
        document = self._document(option, index)
        return QtCore.QSize(option.rect.width(), max(26, int(document.size().height()) + 6))


class ResearchPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()
    reference_levels_changed = QtCore.Signal(bool)
    history_committed = QtCore.Signal(str)
    note_selection_changed = QtCore.Signal(object)
    all_notes_visibility_changed = QtCore.Signal(bool)

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
        notes_header = QtWidgets.QHBoxLayout()
        self.show_all_notes_checkbox = QtWidgets.QCheckBox("顯示全部紀錄")
        self.show_all_notes_checkbox.setChecked(False)
        self.show_all_notes_checkbox.setToolTip("勾選後顯示目前可見範圍內的所有盤中紀錄 Callout；取消後全部隱藏")
        notes_header.addWidget(self.show_all_notes_checkbox)
        notes_header.addStretch(1)
        n_layout.addLayout(notes_header)
        self.note_input = QtWidgets.QTextEdit()
        self.note_input.setPlaceholderText("記錄目前 Replay 時點看到的結構與想法…")
        self.note_input.setMaximumHeight(100)
        self.btn_add_note = QtWidgets.QPushButton("新增盤中紀錄")
        self.note_list = DeselectableNoteListWidget()
        self.note_list.setItemDelegate(IntradayNoteDelegate(self.note_list))
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
        self.show_all_notes_checkbox.toggled.connect(self._all_notes_toggled)
        self.rth_checkbox.toggled.connect(self._rth_toggled)

    def set_case(self, case, replay_time_provider):
        self.case = case
        self.replay_time_provider = replay_time_provider
        self.show_all_notes_checkbox.blockSignals(True)
        self.show_all_notes_checkbox.setChecked(False)
        self.show_all_notes_checkbox.blockSignals(False)
        self.all_notes_visibility_changed.emit(False)
        self.refresh(preserve_note_selection=False)

    def refresh(self, preserve_note_selection=True):
        selected_note_id = None
        if preserve_note_selection:
            current = self._selected_note_item()
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
            widget_item.setData(IntradayNoteDelegate.TIME_ROLE, stamp)
            widget_item.setData(IntradayNoteDelegate.TEXT_ROLE, text)
            self.note_list.addItem(widget_item)
            if selected_note_id is not None and item.get("id") == selected_note_id:
                selected_row = self.note_list.count() - 1
        if selected_row >= 0:
            self.note_list.setCurrentRow(selected_row)
        self.note_list.blockSignals(False)
        self._emit_note_selection()
        if self.show_all_notes_checkbox.isChecked():
            self.all_notes_visibility_changed.emit(True)

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

    def _all_notes_toggled(self, checked: bool):
        # This is a transient chart-display control only; it does not modify Case JSON.
        # Turning it off means "hide all", so disable bulk mode first and then
        # clear any single-note selection without briefly re-rendering all notes.
        self.all_notes_visibility_changed.emit(bool(checked))
        if not checked:
            self.clear_note_selection()

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

    def _selected_note_item(self):
        items = self.note_list.selectedItems()
        return items[0] if items else None

    def clear_note_selection(self):
        """Clear the selected intraday note and therefore its chart callout."""
        if self._editing_note_id is not None:
            self.refresh(preserve_note_selection=False)
            return
        self.note_list.blockSignals(True)
        self.note_list.clearSelection()
        try:
            self.note_list.selectionModel().clearCurrentIndex()
        except Exception:
            pass
        self.note_list.blockSignals(False)
        self.note_selection_changed.emit(None)

    def delete_note(self):
        if self.case is None:
            return
        item = self._selected_note_item()
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
        item = self._selected_note_item()
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
