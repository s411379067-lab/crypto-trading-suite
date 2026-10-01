from __future__ import annotations

import sys
from pathlib import Path
from pyqtgraph.Qt import QtWidgets

from .main_window import PatternViewerWindow
from shared_core.paths import default_case_root


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    root = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_case_root()
    win = PatternViewerWindow(root)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
