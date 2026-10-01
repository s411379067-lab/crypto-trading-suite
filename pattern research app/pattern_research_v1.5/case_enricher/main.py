from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.market_data import MarketDataService
from shared_core.repository import CaseRepository
from shared_core.rth import find_previous_valid_rth


STYLE = """
QWidget { background-color:#0f131c; color:#d7deea; }
QPushButton { background-color:#192131; border:1px solid #4d5a73; border-radius:3px; padding:6px 10px; font-weight:600; }
QPushButton:hover { background-color:#253249; }
QLineEdit, QComboBox, QTimeEdit, QPlainTextEdit {
    background-color:#0d1420; border:1px solid #36445d; border-radius:3px; color:#e6edf7; padding:5px;
}
QCheckBox { spacing:7px; }
QGroupBox { border:1px solid #2a3142; border-radius:4px; margin-top:8px; padding-top:10px; font-weight:700; }
"""


def _weekday_name(d: date) -> str:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d.weekday()]


def _weekend_target(path: Path, is_weekend: bool) -> Path:
    stem = path.stem
    has_suffix = stem.endswith("(W)")
    if is_weekend and not has_suffix:
        return path.with_name(stem + "(W)" + path.suffix)
    if (not is_weekend) and has_suffix:
        return path.with_name(stem[:-3] + path.suffix)
    return path


class EnricherWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Case Enricher v1.5 — Previous RTH")
        self.resize(780, 650)
        self.setStyleSheet(STYLE)
        self.repo = CaseRepository()
        self.market = MarketDataService()

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Case Enricher — Previous RTH High / Low")
        title.setStyleSheet("font-size:16pt; font-weight:700;")
        layout.addWidget(title)

        form = QtWidgets.QFormLayout()
        self.case_root_edit = QtWidgets.QLineEdit(str(Path.cwd() / "cases"))
        self.case_root_btn = QtWidgets.QPushButton("瀏覽")
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.case_root_edit, 1); row.addWidget(self.case_root_btn)
        wrap = QtWidgets.QWidget(); wrap.setLayout(row)
        form.addRow("Case Folder", wrap)

        self.tz_combo = QtWidgets.QComboBox(); self.tz_combo.setEditable(True)
        self.tz_combo.addItems(["America/New_York", "UTC", "Asia/Taipei"])
        self.tz_combo.setCurrentText("America/New_York")
        form.addRow("RTH Session TZ", self.tz_combo)

        self.start_time = QtWidgets.QTimeEdit(QtCore.QTime(9, 30)); self.start_time.setDisplayFormat("HH:mm")
        self.end_time = QtWidgets.QTimeEdit(QtCore.QTime(16, 0)); self.end_time.setDisplayFormat("HH:mm")
        form.addRow("RTH Start", self.start_time)
        form.addRow("RTH End", self.end_time)

        self.only_missing = QtWidgets.QCheckBox("只計算缺少 Previous RTH 的平日 Case")
        self.only_missing.setChecked(True)
        form.addRow("", self.only_missing)
        layout.addLayout(form)

        info = QtWidgets.QLabel(
            "週末 Case：不建立 Previous RTH，寫入 calendar.is_weekend=true，檔名自動加 (W)。\n"
            "平日 Case：往前尋找上一個完整有效 RTH Session；display.timezone 不參與計算。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#9ba8bc;")
        layout.addWidget(info)

        self.run_btn = QtWidgets.QPushButton("開始 Enrich")
        layout.addWidget(self.run_btn)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self.case_root_btn.clicked.connect(self.choose_root)
        self.run_btn.clicked.connect(self.run_enrichment)

    def choose_root(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇 Case Folder", self.case_root_edit.text())
        if selected:
            self.case_root_edit.setText(selected)

    def _append(self, text: str):
        self.log.appendPlainText(text)
        QtWidgets.QApplication.processEvents()

    def run_enrichment(self):
        root = Path(self.case_root_edit.text()).expanduser()
        if not root.exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Case Folder 不存在")
            return
        session_tz = self.tz_combo.currentText().strip() or "America/New_York"
        try:
            ZoneInfo(session_tz)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Error", "RTH Session Timezone 無效")
            return
        start_hhmm = self.start_time.time().toString("HH:mm")
        end_hhmm = self.end_time.time().toString("HH:mm")

        paths = list(self.repo.list_cases(root))
        self.log.clear()
        self._append(f"掃描 {len(paths)} 個 JSON…")
        stats = {"updated": 0, "weekend": 0, "existing": 0, "missing": 0, "error": 0, "skipped": 0}

        self.run_btn.setEnabled(False)
        try:
            for original_path in paths:
                path = Path(original_path)
                try:
                    case = self.repo.load(path)
                    raw_date = str(case.case.get("research_date", "")).strip()
                    case_date = date.fromisoformat(raw_date)
                except Exception as exc:
                    stats["skipped"] += 1
                    self._append(f"SKIP {path.name}: 非 Case JSON / 日期無效 ({exc})")
                    continue

                is_weekend = case_date.weekday() >= 5
                case.calendar.update({
                    "case_date": case_date.isoformat(),
                    "weekday": _weekday_name(case_date),
                    "is_weekend": is_weekend,
                })

                target_path = _weekend_target(path, is_weekend)

                if is_weekend:
                    case.reference_levels.pop("previous_rth", None)
                    case.metadata["rth_enriched_at"] = datetime.now(timezone.utc).isoformat()
                    self.repo.save(path, case)
                    if target_path != path:
                        if target_path.exists():
                            raise FileExistsError(f"rename target already exists: {target_path}")
                        path.rename(target_path)
                        path = target_path
                    stats["weekend"] += 1
                    stats["updated"] += 1
                    self._append(f"WEEKEND {path.name}: 不建立 RTH")
                    continue

                # If a weekday was previously mislabelled with (W), normalize its filename.
                if target_path != path:
                    if target_path.exists():
                        raise FileExistsError(f"rename target already exists: {target_path}")
                    path.rename(target_path)
                    path = target_path

                if self.only_missing.isChecked() and case.reference_levels.get("previous_rth"):
                    self.repo.save(path, case)  # persist calendar/time_context migrations
                    stats["existing"] += 1
                    self._append(f"EXISTS {path.name}: 保留既有 Previous RTH")
                    continue

                try:
                    raw = self.market.load_for_case(path, case.market_data)
                    rth = find_previous_valid_rth(
                        raw,
                        case_date,
                        timezone_name=session_tz,
                        start_hhmm=start_hhmm,
                        end_hhmm=end_hhmm,
                    )
                except Exception as exc:
                    stats["error"] += 1
                    self._append(f"ERROR {path.name}: 行情讀取/計算失敗 ({exc})")
                    continue

                if rth is None:
                    case.reference_levels.pop("previous_rth", None)
                    case.metadata["rth_enriched_at"] = datetime.now(timezone.utc).isoformat()
                    self.repo.save(path, case)
                    stats["missing"] += 1
                    self._append(f"NO DATA {path.name}: 找不到上一個完整 RTH Session")
                    continue

                rth.update({
                    "calculator_version": "1.0",
                    "calculated_at": datetime.now(timezone.utc).isoformat(),
                    "data_source_id": case.market_data.get("source_id", ""),
                })
                case.reference_levels["previous_rth"] = rth
                case.metadata["rth_enriched_at"] = rth["calculated_at"]
                self.repo.save(path, case)
                stats["updated"] += 1
                self._append(
                    f"OK {path.name}: {rth['session_date']}  H={rth['high']:.2f}  L={rth['low']:.2f}"
                )
        finally:
            self.run_btn.setEnabled(True)

        self._append("")
        self._append(
            "完成："
            f" updated={stats['updated']} / weekend={stats['weekend']} / existing={stats['existing']} / "
            f"no_data={stats['missing']} / error={stats['error']} / skipped={stats['skipped']}"
        )


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Calibri", 11))
    win = EnricherWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
