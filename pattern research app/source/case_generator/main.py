from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from shared_core.time_model import build_case_time_range
from case_generator.safe_create import create_case_json_exclusive, find_existing_case_file
from shared_core.paths import default_case_root

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


STYLE = """
QWidget { background-color:#0f131c; color:#d7deea; }
QPushButton { background-color:#192131; border:1px solid #4d5a73; border-radius:3px; padding:6px 10px; font-weight:600; }
QLineEdit, QComboBox, QDateEdit, QTimeEdit {
    background-color:#0d1420; border:1px solid #36445d; border-radius:3px; color:#e6edf7; padding:5px;
}
QGroupBox { border:1px solid #2a3142; border-radius:4px; margin-top:8px; padding-top:10px; font-weight:700; }
"""


class GeneratorWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Case Generator v2.5")
        self.resize(720, 520)
        self.setStyleSheet(STYLE)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Case Generator")
        title.setStyleSheet("font-size:16pt; font-weight:700;")
        layout.addWidget(title)

        form = QtWidgets.QFormLayout()
        self.source_edit = QtWidgets.QLineEdit()
        self.source_btn = QtWidgets.QPushButton("瀏覽")
        src_row = QtWidgets.QHBoxLayout(); src_row.addWidget(self.source_edit, 1); src_row.addWidget(self.source_btn)
        src_wrap = QtWidgets.QWidget(); src_wrap.setLayout(src_row)
        self.symbol_edit = QtWidgets.QLineEdit("NAS100")
        self.tz_combo = QtWidgets.QComboBox(); self.tz_combo.setEditable(True); self.tz_combo.addItems(["Asia/Taipei","America/New_York","UTC"])
        self.start_date = QtWidgets.QDateEdit(QtCore.QDate.currentDate()); self.start_date.setCalendarPopup(True)
        self.end_date = QtWidgets.QDateEdit(QtCore.QDate.currentDate()); self.end_date.setCalendarPopup(True)
        self.a_time = QtWidgets.QTimeEdit(QtCore.QTime(20, 0)); self.a_time.setDisplayFormat("HH:mm")
        self.b_time = QtWidgets.QTimeEdit(QtCore.QTime(21, 30)); self.b_time.setDisplayFormat("HH:mm")
        self.c_time = QtWidgets.QTimeEdit(QtCore.QTime(23, 0)); self.c_time.setDisplayFormat("HH:mm")
        self.view_tf = QtWidgets.QComboBox(); self.view_tf.addItems(["M1","M5","M15","H1"]); self.view_tf.setCurrentText("M5")
        self.output_edit = QtWidgets.QLineEdit(str(default_case_root()))
        self.output_btn = QtWidgets.QPushButton("瀏覽")
        out_row = QtWidgets.QHBoxLayout(); out_row.addWidget(self.output_edit, 1); out_row.addWidget(self.output_btn)
        out_wrap = QtWidgets.QWidget(); out_wrap.setLayout(out_row)

        form.addRow("Market Data", src_wrap)
        form.addRow("Symbol", self.symbol_edit)
        form.addRow("Case Timezone", self.tz_combo)
        form.addRow("Start Date", self.start_date)
        form.addRow("End Date", self.end_date)
        form.addRow("A / data_start", self.a_time)
        form.addRow("B / replay_start", self.b_time)
        form.addRow("C / default_end", self.c_time)
        form.addRow("Default View TF", self.view_tf)
        form.addRow("Output", out_wrap)
        layout.addLayout(form)

        self.generate_btn = QtWidgets.QPushButton("Generate Cases")
        self.safety_note = QtWidgets.QLabel("安全模式：既有 Case 一律跳過，不覆寫研究紀錄")
        self.safety_note.setStyleSheet("color:#91c9ff; font-weight:600;")
        self.result = QtWidgets.QPlainTextEdit(); self.result.setReadOnly(True)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.safety_note)
        layout.addWidget(self.result, 1)

        self.source_btn.clicked.connect(self.choose_source)
        self.output_btn.clicked.connect(self.choose_output)
        self.generate_btn.clicked.connect(self.generate)

    def choose_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Market Data", "", "Data (*.txt *.tsv *.csv *.parquet)")
        if path: self.source_edit.setText(path)

    def choose_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output Folder", self.output_edit.text())
        if path: self.output_edit.setText(path)

    def generate(self):
        source = Path(self.source_edit.text()).expanduser()
        if not source.exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Market Data 檔案不存在")
            return
        out_root = Path(self.output_edit.text()).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)
        symbol = self.symbol_edit.text().strip() or "UNKNOWN"
        tz_name = self.tz_combo.currentText().strip() or "Asia/Taipei"
        try:
            ZoneInfo(tz_name)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Error", "Timezone 無效")
            return

        d0 = self.start_date.date()
        d1 = self.end_date.date()
        if d1 < d0:
            d0, d1 = d1, d0
        count = d0.daysTo(d1) + 1
        created: list[Path] = []
        skipped: list[Path] = []
        for i in range(count):
            day = d0.addDays(i)
            day_str = day.toString("yyyy-MM-dd")
            folder = out_root / symbol / day.toString("yyyy") / day.toString("MM")

            # Structural safety rule: an existing Case for the same research
            # date is never regenerated.  Weekend files renamed by Enricher to
            # YYYY-MM-DD(W).json are treated as the same Case date.
            existing = find_existing_case_file(folder, day_str)
            if existing is not None:
                skipped.append(existing)
                continue

            py_day = date(day.year(), day.month(), day.day())
            tr = build_case_time_range(
                py_day,
                time(self.a_time.time().hour(), self.a_time.time().minute(), self.a_time.time().second()),
                time(self.b_time.time().hour(), self.b_time.time().minute(), self.b_time.time().second()),
                time(self.c_time.time().hour(), self.c_time.time().minute(), self.c_time.time().second()),
                tz_name,
            )
            case = {
                "schema_version": "0.1",
                "case": {"id": f"case-{uuid.uuid4().hex[:12]}", "symbol": symbol, "research_date": day_str},
                "market_data": {
                    "source_id": f"{symbol.lower()}-primary",
                    "source_type": source.suffix.lower().lstrip("."),
                    "source_path": str(source.resolve()),
                    "resolution": "M1"
                },
                "time_range": tr,
                "time_context": {
                    "case_timezone": tz_name,
                    "canonical_timezone": "UTC"
                },
                "display": {"view_timeframe": self.view_tf.currentText(), "timezone": tz_name, "x_tick_interval": "15m"},
                "replay": {"step_minutes": 1, "current_time": tr["replay_start"]},
                "calendar": {}, "reference_levels": {}, "reference_statistics": {},
                "patterns": [], "intraday_notes": [], "drawings": [],
                "metadata": {"created_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(), "updated_at": datetime.now(tz=ZoneInfo("UTC")).isoformat()}
            }
            path = folder / f"{day_str}.json"
            if create_case_json_exclusive(path, case):
                created.append(path)
            else:
                # Race-safe fallback: another process created the destination
                # after the initial check.  It is still never overwritten.
                skipped.append(path)

        lines = [
            f"完成：created={len(created)} / skipped existing={len(skipped)}",
            "",
        ]
        lines.extend(f"[CREATE] {p}" for p in created)
        lines.extend(f"[SKIP]   {p}" for p in skipped)
        self.result.setPlainText("\n".join(lines))


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Calibri", 11))
    win = GeneratorWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
