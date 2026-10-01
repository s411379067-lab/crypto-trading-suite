from __future__ import annotations

import sys
from pyqtgraph.Qt import QtGui, QtWidgets

from .main_window import MainWindow
from shared_core.paths import default_case_root


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Calibri", 11))
    default_root = default_case_root()
    default_root.mkdir(parents=True, exist_ok=True)
    win = MainWindow(default_root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
