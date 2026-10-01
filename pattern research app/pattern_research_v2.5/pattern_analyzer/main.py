from __future__ import annotations

import sys
from pathlib import Path
from pyqtgraph.Qt import QtGui, QtWidgets

from .main_window import MainWindow


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Calibri", 11))
    default_root = Path.cwd() / "cases"
    default_root.mkdir(parents=True, exist_ok=True)
    win = MainWindow(default_root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
