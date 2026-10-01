from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from shared_core.market_data import MarketDataService
from shared_core.repository import CaseRepository
from shared_core.rth import (
    calculate_intraday_volatility,
    find_previous_valid_rth,
    intraday_volatility_is_current,
    previous_rth_is_current,
)


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


RTH_CALCULATOR_VERSION = "1.1"
VOL_CALCULATOR_VERSION = "1.0"
VOL_SESSION_COUNT = 20


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
        self.setWindowTitle("Case Enricher v1.19 — RTH + 20D Intraday Volatility")
        self.resize(860, 760)
        self.setStyleSheet(STYLE)
        self.repo = CaseRepository()
        self.market = MarketDataService()

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Case Enricher — RTH Reference + 20D Intraday Volatility")
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
        layout.addLayout(form)

        prev_box = QtWidgets.QGroupBox("Previous RTH H/L/C")
        prev_layout = QtWidgets.QVBoxLayout(prev_box)
        self.only_missing = QtWidgets.QCheckBox("只計算缺少／過期 Previous RTH H/L/C 的平日 Case")
        self.only_missing.setChecked(True)
        prev_layout.addWidget(self.only_missing)
        layout.addWidget(prev_box)

        vol_box = QtWidgets.QGroupBox("20D RTH Intraday Volatility")
        vol_layout = QtWidgets.QVBoxLayout(vol_box)
        self.vol_enabled = QtWidgets.QCheckBox("計算最近 20 個完整 RTH Session 的日內波動統計")
        self.vol_enabled.setChecked(True)
        self.vol_only_missing = QtWidgets.QCheckBox("只計算缺少／過期的 20D 波動統計")
        self.vol_only_missing.setChecked(True)
        vol_layout.addWidget(self.vol_enabled)
        vol_layout.addWidget(self.vol_only_missing)
        desc = QtWidgets.QLabel(
            "Range Points = RTH High - RTH Low；Range % = (High - Low) / RTH Open × 100；"
            "Std 使用樣本標準差 ddof=1。這版只寫入 JSON / Enricher 顯示，不在 Chart 畫波動線。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#9ba8bc;")
        vol_layout.addWidget(desc)
        layout.addWidget(vol_box)

        summary_box = QtWidgets.QGroupBox("20D 波動統計 — 最新處理 Case")
        summary = QtWidgets.QGridLayout(summary_box)
        self.summary_case = QtWidgets.QLabel("—")
        self.summary_sessions = QtWidgets.QLabel("—")
        self.summary_med_pts = QtWidgets.QLabel("—")
        self.summary_std_pts = QtWidgets.QLabel("—")
        self.summary_med_pct = QtWidgets.QLabel("—")
        self.summary_std_pct = QtWidgets.QLabel("—")
        value_style = "font-weight:700; color:#f0f5ff;"
        for w in (self.summary_case, self.summary_sessions, self.summary_med_pts, self.summary_std_pts,
                  self.summary_med_pct, self.summary_std_pct):
            w.setStyleSheet(value_style)
        summary.addWidget(QtWidgets.QLabel("Case"), 0, 0); summary.addWidget(self.summary_case, 0, 1, 1, 3)
        summary.addWidget(QtWidgets.QLabel("Sessions"), 1, 0); summary.addWidget(self.summary_sessions, 1, 1)
        summary.addWidget(QtWidgets.QLabel("Median Points"), 2, 0); summary.addWidget(self.summary_med_pts, 2, 1)
        summary.addWidget(QtWidgets.QLabel("Std Points"), 2, 2); summary.addWidget(self.summary_std_pts, 2, 3)
        summary.addWidget(QtWidgets.QLabel("Median %"), 3, 0); summary.addWidget(self.summary_med_pct, 3, 1)
        summary.addWidget(QtWidgets.QLabel("Std %"), 3, 2); summary.addWidget(self.summary_std_pct, 3, 3)
        layout.addWidget(summary_box)

        info = QtWidgets.QLabel(
            "週末 Case：不建立 Previous RTH，檔名自動加 (W)；但 20D 波動統計仍使用週末前的 20 個完整 RTH Session。\n"
            "平日 Case：Previous RTH 與 20D 波動統計都以 RTH Session TZ/時段計算；display.timezone 不參與計算。"
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
        self.vol_enabled.toggled.connect(self.vol_only_missing.setEnabled)

    def choose_root(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇 Case Folder", self.case_root_edit.text())
        if selected:
            self.case_root_edit.setText(selected)

    def _append(self, text: str):
        self.log.appendPlainText(text)
        QtWidgets.QApplication.processEvents()

    def _show_vol_summary(self, case_name: str, vol: dict | None):
        self.summary_case.setText(case_name)
        if not isinstance(vol, dict):
            self.summary_sessions.setText("無完整 20D 資料")
            self.summary_med_pts.setText("—")
            self.summary_std_pts.setText("—")
            self.summary_med_pct.setText("—")
            self.summary_std_pct.setText("—")
            return
        sessions = vol.get("sessions", [])
        if sessions:
            self.summary_sessions.setText(f"{sessions[0]['date']} → {sessions[-1]['date']} ({len(sessions)})")
        else:
            self.summary_sessions.setText(str(vol.get("observations", 0)))
        self.summary_med_pts.setText(f"{float(vol['median_range_points']):.2f}")
        self.summary_std_pts.setText(f"{float(vol['std_range_points']):.2f}")
        self.summary_med_pct.setText(f"{float(vol['median_range_pct']):.3f}%")
        self.summary_std_pct.setText(f"{float(vol['std_range_pct']):.3f}%")

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
        counts = {
            "rth_updated": 0, "rth_existing": 0, "rth_missing": 0,
            "vol_updated": 0, "vol_existing": 0, "vol_missing": 0,
            "weekend": 0, "error": 0, "skipped": 0,
        }

        self.run_btn.setEnabled(False)
        try:
            for original_path in paths:
                path = Path(original_path)
                try:
                    case = self.repo.load(path)
                    raw_date = str(case.case.get("research_date", "")).strip()
                    case_date = date.fromisoformat(raw_date)
                except Exception as exc:
                    counts["skipped"] += 1
                    self._append(f"SKIP {path.name}: 非 Case JSON / 日期無效 ({exc})")
                    continue

                is_weekend = case_date.weekday() >= 5
                case.calendar.update({
                    "case_date": case_date.isoformat(),
                    "weekday": _weekday_name(case_date),
                    "is_weekend": is_weekend,
                })

                target_path = _weekend_target(path, is_weekend)
                if target_path != path:
                    if target_path.exists():
                        counts["error"] += 1
                        self._append(f"ERROR {path.name}: rename target already exists: {target_path}")
                        continue
                    path.rename(target_path)
                    path = target_path

                if is_weekend:
                    case.reference_levels.pop("previous_rth", None)
                    counts["weekend"] += 1

                source_id = str(case.market_data.get("source_id", ""))
                existing_rth = case.reference_levels.get("previous_rth")
                rth_current = (not is_weekend) and previous_rth_is_current(
                    existing_rth,
                    calculator_version=RTH_CALCULATOR_VERSION,
                    timezone_name=session_tz,
                    start_hhmm=start_hhmm,
                    end_hhmm=end_hhmm,
                    data_source_id=source_id,
                )

                existing_vol = case.reference_statistics.get("intraday_volatility_20d")
                vol_current = intraday_volatility_is_current(
                    existing_vol,
                    calculator_version=VOL_CALCULATOR_VERSION,
                    session_count=VOL_SESSION_COUNT,
                    timezone_name=session_tz,
                    start_hhmm=start_hhmm,
                    end_hhmm=end_hhmm,
                    data_source_id=source_id,
                )

                need_rth = (not is_weekend) and (not self.only_missing.isChecked() or not rth_current)
                need_vol = self.vol_enabled.isChecked() and (not self.vol_only_missing.isChecked() or not vol_current)

                raw = None
                if need_rth or need_vol:
                    try:
                        raw = self.market.load_for_case(path, case.market_data)
                    except Exception as exc:
                        counts["error"] += 1
                        self._append(f"ERROR {path.name}: 行情讀取失敗 ({exc})")
                        self.repo.save(path, case)
                        continue

                if is_weekend:
                    self._append(f"WEEKEND {path.name}: 不建立 Previous RTH")
                elif not need_rth:
                    counts["rth_existing"] += 1
                    self._append(f"RTH EXISTS {path.name}: H/L/C 已完整，版本={RTH_CALCULATOR_VERSION}")
                else:
                    if existing_rth and not rth_current:
                        self._append(f"RTH UPGRADE {path.name}: 既有資料缺欄位或版本/設定過期")
                    try:
                        rth = find_previous_valid_rth(
                            raw,
                            case_date,
                            timezone_name=session_tz,
                            start_hhmm=start_hhmm,
                            end_hhmm=end_hhmm,
                        )
                    except Exception as exc:
                        rth = None
                        self._append(f"RTH ERROR {path.name}: {exc}")
                    if rth is None:
                        case.reference_levels.pop("previous_rth", None)
                        counts["rth_missing"] += 1
                        self._append(f"RTH NO DATA {path.name}: 找不到上一個完整 RTH Session")
                    else:
                        rth.update({
                            "calculator_version": RTH_CALCULATOR_VERSION,
                            "calculated_at": datetime.now(timezone.utc).isoformat(),
                            "data_source_id": source_id,
                        })
                        case.reference_levels["previous_rth"] = rth
                        case.metadata["rth_enriched_at"] = rth["calculated_at"]
                        counts["rth_updated"] += 1
                        self._append(
                            f"RTH OK {path.name}: {rth['session_date']}  H={rth['high']:.2f}  L={rth['low']:.2f}  C={rth['close']:.2f}"
                        )

                if not self.vol_enabled.isChecked():
                    pass
                elif not need_vol:
                    counts["vol_existing"] += 1
                    self._show_vol_summary(path.name, existing_vol)
                    self._append(
                        f"VOL EXISTS {path.name}: Median={float(existing_vol['median_range_points']):.2f} pts / "
                        f"Std={float(existing_vol['std_range_points']):.2f} pts / "
                        f"Median={float(existing_vol['median_range_pct']):.3f}% / Std={float(existing_vol['std_range_pct']):.3f}%"
                    )
                else:
                    if existing_vol and not vol_current:
                        self._append(f"VOL UPGRADE {path.name}: 既有 20D 統計版本/設定過期")
                    try:
                        vol = calculate_intraday_volatility(
                            raw,
                            case_date,
                            session_count=VOL_SESSION_COUNT,
                            timezone_name=session_tz,
                            start_hhmm=start_hhmm,
                            end_hhmm=end_hhmm,
                            lookback_days=60,
                        )
                    except Exception as exc:
                        vol = None
                        self._append(f"VOL ERROR {path.name}: {exc}")
                    if vol is None:
                        case.reference_statistics.pop("intraday_volatility_20d", None)
                        counts["vol_missing"] += 1
                        self._show_vol_summary(path.name, None)
                        self._append(f"VOL NO DATA {path.name}: 60 個日曆日內不足 {VOL_SESSION_COUNT} 個完整 RTH Session")
                    else:
                        vol.update({
                            "calculator_version": VOL_CALCULATOR_VERSION,
                            "calculated_at": datetime.now(timezone.utc).isoformat(),
                            "data_source_id": source_id,
                        })
                        case.reference_statistics["intraday_volatility_20d"] = vol
                        case.metadata["intraday_volatility_20d_enriched_at"] = vol["calculated_at"]
                        counts["vol_updated"] += 1
                        self._show_vol_summary(path.name, vol)
                        self._append(
                            f"VOL OK {path.name}: Median={vol['median_range_points']:.2f} pts  "
                            f"Std={vol['std_range_points']:.2f} pts  "
                            f"Median={vol['median_range_pct']:.3f}%  Std={vol['std_range_pct']:.3f}%"
                        )

                self.repo.save(path, case)
        finally:
            self.run_btn.setEnabled(True)

        self._append("")
        self._append(
            "完成："
            f" RTH updated={counts['rth_updated']} / existing={counts['rth_existing']} / no_data={counts['rth_missing']}；"
            f" VOL updated={counts['vol_updated']} / existing={counts['vol_existing']} / no_data={counts['vol_missing']}；"
            f" weekend={counts['weekend']} / error={counts['error']} / skipped={counts['skipped']}"
        )


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Calibri", 11))
    win = EnricherWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
