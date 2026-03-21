"""
Simple Daily Regime Analyzer
下載 1h 資料後，可依指定 session（all day / asian）聚合成日線，
計算 EMA20 / is_bull / body_ratio，最後可查詢指定日期。
"""

import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
from PyQt5.QtCore import QDate, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
ASIAN_START_HOUR = 0
ASIAN_END_HOUR = 10
TIMEFRAME = "1h"
LIMIT = 1000


class DataFetchThread(QThread):
    progress = pyqtSignal(str)
    data_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, coin="BTC", months=3):
        super().__init__()
        self.coin = coin
        self.months = months

    def run(self):
        try:
            self.progress.emit(f"正在下載 {self.coin} 資料...")

            exchange = ccxt.binance({"enableRateLimit": True})
            symbol = f"{self.coin}/USDT"

            since_dt = datetime.now(timezone.utc) - timedelta(days=30 * self.months)
            since_ms = int(since_dt.timestamp() * 1000)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            rows = []
            batch_count = 0

            while True:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=TIMEFRAME,
                    since=since_ms,
                    limit=LIMIT,
                )

                if not batch:
                    break

                rows.extend(batch)
                batch_count += 1
                self.progress.emit(
                    f"已下載 {batch_count} 批，共 {len(rows)} 根 1h K線"
                )

                last_ts = batch[-1][0]
                next_since_ms = last_ts + 1

                if next_since_ms <= since_ms:
                    break

                since_ms = next_since_ms

                if last_ts >= now_ms - exchange.parse_timeframe(TIMEFRAME) * 1000:
                    break

                time.sleep(max(exchange.rateLimit / 1000.0, 0.2))

            if not rows:
                raise ValueError("查無可用資料")

            df = pd.DataFrame(
                rows,
                columns=["ts", "open", "high", "low", "close", "vol"],
            ).drop_duplicates("ts")

            df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.sort_values("dt_utc").reset_index(drop=True)

            self.data_ready.emit(df)
            self.progress.emit(f"✅ 下載完成，共 {len(df)} 根 1h K線")

        except Exception as exc:
            self.error.emit(f"❌ 下載失敗: {exc}")


class DailyRegimeAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df_hourly = pd.DataFrame()
        self.df_daily = pd.DataFrame()
        self.fetch_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Simple Daily Regime Analyzer")
        self.setGeometry(200, 120, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_group = QGroupBox("設定")
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("幣種"))
        self.coin_combo = QComboBox()
        self.coin_combo.addItems(["BTC", "ETH", "SOL", "BNB"])
        control_layout.addWidget(self.coin_combo)

        control_layout.addWidget(QLabel("回溯月數"))
        self.months_spin = QSpinBox()
        self.months_spin.setRange(1, 24)
        self.months_spin.setValue(3)
        control_layout.addWidget(self.months_spin)

        control_layout.addWidget(QLabel("Session"))
        self.session_combo = QComboBox()
        self.session_combo.addItem("all day", "all_day")
        self.session_combo.addItem("asian", "asian")
        self.session_combo.currentIndexChanged.connect(self.on_session_changed)
        control_layout.addWidget(self.session_combo)

        self.btn_download = QPushButton("下載資料")
        self.btn_download.clicked.connect(self.download_data)
        control_layout.addWidget(self.btn_download)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        query_group = QGroupBox("日期查詢")
        query_layout = QHBoxLayout()

        query_layout.addWidget(QLabel("日期"))
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate.currentDate())
        query_layout.addWidget(self.date_edit)

        self.btn_query = QPushButton("查詢")
        self.btn_query.setEnabled(False)
        self.btn_query.clicked.connect(self.query_date)
        query_layout.addWidget(self.btn_query)

        query_group.setLayout(query_layout)
        main_layout.addWidget(query_group)

        self.status_label = QLabel("請先下載資料")
        main_layout.addWidget(self.status_label)

        recent_group = QGroupBox("近期日線資料")
        recent_layout = QVBoxLayout()
        self.recent_text = QTextEdit()
        self.recent_text.setReadOnly(True)
        recent_layout.addWidget(self.recent_text)
        recent_group.setLayout(recent_layout)
        main_layout.addWidget(recent_group)

        result_group = QGroupBox("查詢結果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

    def download_data(self):
        self.btn_download.setEnabled(False)
        self.btn_query.setEnabled(False)
        self.status_label.setText("下載中...")
        self.recent_text.clear()
        self.result_text.clear()

        coin = self.coin_combo.currentText()
        months = self.months_spin.value()

        self.fetch_thread = DataFetchThread(coin=coin, months=months)
        self.fetch_thread.progress.connect(self.status_label.setText)
        self.fetch_thread.data_ready.connect(self.on_data_ready)
        self.fetch_thread.error.connect(self.on_data_error)
        self.fetch_thread.start()

    def on_data_error(self, msg):
        self.status_label.setText(msg)
        self.result_text.setText(msg)
        self.btn_download.setEnabled(True)
        self.btn_query.setEnabled(False)

    def get_selected_session(self):
        return self.session_combo.currentData() or "all_day"

    def build_daily_dataframe(self, df):
        if df.empty:
            return pd.DataFrame()

        work = df.copy()
        work = work.sort_values("dt_utc").reset_index(drop=True)

        session = self.get_selected_session()
        if session == "asian":
            mask = (
                (work["dt_utc"].dt.hour >= ASIAN_START_HOUR)
                & (work["dt_utc"].dt.hour < ASIAN_END_HOUR)
            )
            work = work.loc[mask].copy()

        if work.empty:
            return pd.DataFrame()

        work["date"] = work["dt_utc"].dt.date
        work = work.sort_values(["date", "dt_utc"]).reset_index(drop=True)
        work["bar_index"] = work.groupby("date").cumcount() + 1

        daily = work.groupby("date", as_index=False).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            vol=("vol", "sum"),
            bar_count=("bar_index", "max"),
        )

        daily = daily.sort_values("date").reset_index(drop=True)

        rng = (daily["high"] - daily["low"]).replace(0, pd.NA)
        daily["ema20"] = daily["close"].ewm(span=20, adjust=False).mean()
        daily["is_bull"] = daily["close"] > daily["open"]
        daily["above_ema20"] = daily["close"] > daily["ema20"]
        daily["body_ratio"] = (
            (daily["close"] - daily["open"]).abs() / rng
        ).fillna(0.0)

        return daily

    def on_session_changed(self):
        if self.df_hourly.empty:
            return

        self.df_daily = self.build_daily_dataframe(self.df_hourly)
        if self.df_daily.empty:
            self.status_label.setText("❌ 沒有可用的日線資料")
            self.btn_query.setEnabled(False)
            self.recent_text.clear()
            self.result_text.clear()
            return

        self.btn_query.setEnabled(True)
        session_label = "亞洲盤" if self.get_selected_session() == "asian" else "全日盤"
        self.status_label.setText(
            f"✅ 已切換為 {session_label}，共 {len(self.df_daily)} 天日線資料"
        )
        self.show_recent_data()

    def on_data_ready(self, df):
        self.df_hourly = df
        self.df_daily = self.build_daily_dataframe(df)

        self.btn_download.setEnabled(True)

        if self.df_daily.empty:
            self.status_label.setText("❌ 沒有可用的日線資料")
            self.btn_query.setEnabled(False)
            return

        self.btn_query.setEnabled(True)
        session_label = "亞洲盤" if self.get_selected_session() == "asian" else "全日盤"
        self.status_label.setText(
            f"✅ 已載入 {session_label}資料，共 {len(self.df_daily)} 天日線資料"
        )
        self.show_recent_data()

        last_date = self.df_daily["date"].iloc[-1]
        self.date_edit.setDate(QDate(last_date.year, last_date.month, last_date.day))

    def show_recent_data(self):
        cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "ema20",
            "is_bull",
            "above_ema20",
            "body_ratio",
        ]
        recent_df = self.df_daily[cols].tail(10).copy()
        self.recent_text.setText(recent_df.to_string(index=False))

    def query_date(self):
        if self.df_daily.empty:
            self.result_text.setText("請先下載資料")
            return

        query_date = self.date_edit.date().toPyDate()
        result = self.df_daily[self.df_daily["date"] == query_date]

        if result.empty:
            self.result_text.setText(f"❌ 找不到 {query_date} 的資料")
            return

        row = result.iloc[0]
        range_size = row["high"] - row["low"]
        range_pct = (range_size / row["open"] * 100) if row["open"] else 0

        lines = [
            f"日期: {query_date}, O: {row['open']:.2f}, H: {row['high']:.2f}, L: {row['low']:.2f}, C: {row['close']:.2f}, Volume: {row['vol']:.2f}, 波動: {range_size:.2f} ({range_pct:.2f}%)",
        ]

        session_label = "亞洲盤" if self.get_selected_session() == "asian" else "全日盤"
        lines.append(f"{session_label}K線數: {int(row['bar_count'])}")

        if pd.notna(row["ema20"]):
            lines.append(f"EMA20: {row['ema20']:.2f}")
            lines.append(f"與EMA20關係: {'在EMA20上方 🟢' if row['above_ema20'] else '在EMA20下方 🔴'}")
            lines.append(f"K線方向: {'陽線 🟢' if row['is_bull'] else '陰線 🔴'}")
        else:
            lines.append("EMA20: N/A")
            lines.append("與EMA20關係: N/A")
            lines.append(f"K線方向: {'陽線 🟢' if row['is_bull'] else '陰線 🔴'}")

        self.result_text.setText("\n".join(lines))

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        window = DailyRegimeAnalyzerUI()
        window.show()
        sys.exit(app.exec_())

    window = DailyRegimeAnalyzerUI()
    window.show()
    return window


if __name__ == "__main__":
    main()