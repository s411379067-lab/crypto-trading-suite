from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets


class ResearchPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.case = None
        self.replay_time_provider = lambda: ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Pattern Research")
        title.setStyleSheet("font-size:13pt; font-weight:700; color:#e6edf7;")
        layout.addWidget(title)

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

    def set_case(self, case, replay_time_provider):
        self.case = case
        self.replay_time_provider = replay_time_provider
        self.refresh()

    def refresh(self):
        self.pattern_list.clear()
        self.note_list.clear()
        if self.case is None:
            return
        for item in self.case.patterns:
            widget_item = QtWidgets.QListWidgetItem(item.get("text", ""))
            widget_item.setData(QtCore.Qt.UserRole, item.get("id"))
            self.pattern_list.addItem(widget_item)
        for item in self.case.intraday_notes:
            stamp = item.get("replay_time", "")
            text = item.get("text", "")
            widget_item = QtWidgets.QListWidgetItem(f"{stamp}\n{text}")
            widget_item.setData(QtCore.Qt.UserRole, item.get("id"))
            self.note_list.addItem(widget_item)

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
