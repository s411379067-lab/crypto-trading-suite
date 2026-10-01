from __future__ import annotations

from pathlib import Path
from pyqtgraph.Qt import QtCore, QtWidgets


class CaseLibraryWidget(QtWidgets.QWidget):
    case_open_requested = QtCore.Signal(str)
    root_changed = QtCore.Signal(str)

    def __init__(self, root_dir: str | Path, parent=None):
        super().__init__(parent)
        self.root_dir = Path(root_dir).resolve()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("Case Library")
        title.setStyleSheet("font-size:13pt; font-weight:700; color:#e6edf7;")
        layout.addWidget(title)

        self.btn_root = QtWidgets.QPushButton("選擇 Case 資料夾")
        self.btn_root.clicked.connect(self.choose_root)
        layout.addWidget(self.btn_root)

        self.path_label = QtWidgets.QLabel(str(self.root_dir))
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color:#8f9bad; font-size:9pt;")
        layout.addWidget(self.path_label)

        self.model = QtWidgets.QFileSystemModel(self)
        self.model.setNameFilters(["*.json"])
        self.model.setNameFilterDisables(False)
        self.model.setRootPath(str(self.root_dir))

        self.tree = QtWidgets.QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(str(self.root_dir)))
        self.tree.setHeaderHidden(True)
        for col in range(1, 4):
            self.tree.hideColumn(col)
        self.tree.doubleClicked.connect(self._double_clicked)
        layout.addWidget(self.tree, 1)

    def choose_root(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇 Case Library", str(self.root_dir))
        if not selected:
            return
        self.set_root(selected)

    def set_root(self, root: str | Path):
        self.root_dir = Path(root).resolve()
        self.model.setRootPath(str(self.root_dir))
        self.tree.setRootIndex(self.model.index(str(self.root_dir)))
        self.path_label.setText(str(self.root_dir))
        self.root_changed.emit(str(self.root_dir))

    def _double_clicked(self, index):
        path = Path(self.model.filePath(index))
        if path.is_file() and path.suffix.lower() == ".json":
            self.case_open_requested.emit(str(path))
