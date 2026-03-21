import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyarrow.parquet as pq


# ================================================================
# 自訂時間軸（顯示 HH:MM (MM.DD)）
# ================================================================
class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        labels = []
        for ts in values:
            try:
                dt = pd.to_datetime(ts, unit="s").tz_localize("UTC").tz_convert("America/New_York")

                minute = dt.minute
                if minute not in (0, 15, 30, 45):
                    labels.append("")
                    continue

                time_str = dt.strftime("%H:%M")
                md_str = dt.strftime("%Y.%m.%d")
                labels.append(f"{time_str}\n({md_str})")
            except Exception:
                labels.append("")
        return labels


# ================================================================
# 主程式
# ================================================================
def main():
    START_TIME = "2025-12-31 05:40"


    N_minutes = 5  # K 線時間週期
    max_SL = 60  # 最大停損點數
    N_R_SL = 1  # 以 R 倍數計算停損
    line_mode = False
    line_start = None
    temp_line = None
    grid_lines = []
    saved_lines = []  # 保存所有劃線
    fibo_groups = []  # 保存所有斐波那契群組
    selected_item = None  # 當前選中的圖形項（線或文字）
    line_custom_pens = {}  # 存儲每條線的自訂顏色 {line_obj: pen}
    fibo_mode = False  # 斐波那契模式
    fibo_base_price = None  # 斐波那契基準價格
    text_mode = False  # 打字模式
    range_mode = False  # 價格範圍測量模式
    range_start_price = None  # 價格範圍起點
    price_ranges = []  # 保存所有價格範圍物件
    coin_name = "ETH"

    # ----------------------------- 讀取 parquet -----------------------------
    # df = pq.read_table("ETH_1m_6M_UTC.parquet").to_pandas()
    df = pq.read_table(fr"data source/{coin_name}_{N_minutes}m_48M_UTC.parquet").to_pandas()
    
    df["dt_ny"] = pd.to_datetime(df["dt_utc"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["dt_ny"].dt.date
    df = df.sort_values(["date", "dt_ny"]).reset_index(drop=True)
    df["bar_index"] = df.groupby("date").cumcount() + 1
    df["MA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["MA100"] = df["close"].ewm(span=100, adjust=False).mean()

    # 價格精度：以當日第一根開盤價判斷，<10 用 4 位，>=10 用 2 位
    df["day_open_price"] = df.groupby("date")["open"].transform("first")
    df["price_precision"] = df["day_open_price"].apply(lambda v: 4 if v < 10 else 2).astype(int)
    price_precision_default = int(df["price_precision"].iloc[0]) if len(df) > 0 else 2

    df["dt_ny_ts"] = df["dt_ny"].map(lambda x: x.timestamp())

    ## 只取指定時間前後3天的資料
    start_dt = pd.to_datetime(START_TIME).tz_localize("America/New_York")
    df = df[(df["dt_ny"] >= start_dt - pd.Timedelta(days=3)) & (df["dt_ny"] <= start_dt + pd.Timedelta(days=3))]
    df = df.reset_index(drop=True)

    # ================================================================
    # ⭐ 新增 Body Ratio / ΔC 計算在 df
    # ================================================================
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"])
    df["body_ratio"] = 100*df["body_ratio"].replace([float('inf'), -float('inf')], 0).fillna(0)
    df["body_ratio_level"] = pd.cut(df["body_ratio"], bins=[-1, 34, 68, 101], labels=['L', 'M', 'H'])
    df["K_range"] = (df["high"] - df["low"])*100/df["open"]
    df["K_range_level"] = pd.cut(df["K_range"], bins=[-1, 0.35, 0.6, 100], labels=['S', 'M', 'L'])

    # K1 C2_profit_R calculation
    # if K0 is bear bar C2_profit_R = (K0_low - K1_close)/K0_K_range
    # if K0 bull bar C2_profit_R = (K1_close - K0_high)/K0_K_range
    df["C2_profit_R"] = 0.0
    df["Max_K2_profit_R"] = 0.0
    df["0.4R"] = 0.0
    df["0.7R"] = 0.0
    df["1.0R"] = 0.0
    df["1.3R"] = 0.0
    df["1.6R"] = 0.0
    df["2.0R"] = 0.0
    df["SL_price"] = 0.0
    df["entry_price"] = 0.0
    df["size"] = 0.0
    df["shadow_ratio"] = 0
    df["counter_shadow_ratio"] = 0
    for i in range(len(df)):
        if i != 0:
            if df.at[i-1, "close"] < df.at[i-1, "open"]:  # K0 is bear bar
                df.at[i, "C2_profit_R"] = (df.at[i-1, "low"] - df.at[i, "close"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
                df.at[i, "Max_K2_profit_R"] = (df.at[i-1, "low"] - df.at[i, "low"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
            else:  # K0 is bull bar
                df.at[i, "C2_profit_R"] = (df.at[i, "close"] - df.at[i-1, "high"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
                df.at[i, "Max_K2_profit_R"] = (df.at[i, "high"] - df.at[i-1, "high"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
        if df.at[i, "close"] < df.at[i, "open"]:  # K0 is bear bar
            df.at[i, "0.4R"] = df.at[i, "low"] - 0.4 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "0.7R"] = df.at[i, "low"] - 0.7 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "1.0R"] = df.at[i, "low"] - 1.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "1.3R"] = df.at[i, "low"] - 1.3 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "1.6R"] = df.at[i, "low"] - 1.6 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "2.0R"] = df.at[i, "low"] - 2.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "SL_price"] = df.at[i, "low"] + N_R_SL * (df.at[i, "high"] - df.at[i, "low"])
            df.at[i, "entry_price"] = df.at[i, "low"]
            df.at[i, "size"] = max_SL / (N_R_SL * (df.at[i, "high"] - df.at[i, "low"]))
            df.at[i, "shadow_ratio"] = (df.at[i, "high"] - df.at[i, "open"])*100/(df.at[i, "high"] - df.at[i, "low"])
            df.at[i, "counter_shadow_ratio"] = (df.at[i, "close"] - df.at[i, "low"])*100/(df.at[i, "high"] - df.at[i, "low"])
            
        else:  # K0 is bull bar
            df.at[i, "0.4R"] = df.at[i, "high"] + 0.4 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "0.7R"] = df.at[i, "high"] + 0.7 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "1.0R"] = df.at[i, "high"] + 1.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "1.3R"] = df.at[i, "high"] + 1.3 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "1.6R"] = df.at[i, "high"] + 1.6 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "2.0R"] = df.at[i, "high"] + 2.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
            df.at[i, "SL_price"] = df.at[i, "high"] - N_R_SL * (df.at[i, "high"] - df.at[i, "low"])
            df.at[i, "entry_price"] = df.at[i, "high"]
            df.at[i, "size"] = max_SL / (N_R_SL * (df.at[i, "high"] - df.at[i, "low"]))
            df.at[i, "shadow_ratio"] = (df.at[i, "open"] - df.at[i, "low"])*100/(df.at[i, "high"] - df.at[i, "low"])
            df.at[i, "counter_shadow_ratio"] = (df.at[i, "high"] - df.at[i, "close"])*100/(df.at[i, "high"] - df.at[i, "low"])
        
    

    df["delta_c"] = df["close"].diff().fillna(0)*100/df["close"]


    # ================================================================
    # ⭐⭐ 啟動時設定起始時間
    # ================================================================

    start_dt = pd.to_datetime(START_TIME).tz_localize("America/New_York")
    start_idx = df[df["dt_ny"] >= start_dt].index.min()

    if pd.isna(start_idx):
        raise ValueError(f"START_TIME {START_TIME} 超出資料範圍")

    df = df.loc[start_idx:].reset_index(drop=True)
    df["bar_index"] = df.index

    bars = df.to_dict("records")
    total = len(bars)
    idx = 1


    # ================================================================
    # 建立 QApplication
    # ================================================================
    app = QtWidgets.QApplication([])

    # ================================================================
    # 建立 UI
    # ================================================================
    # 頂層視窗，左側為圖表，右側為下單面板
    root = QtWidgets.QWidget()
    root.setWindowTitle(f"{coin_name} Replay UI 5m")
    hbox = QtWidgets.QHBoxLayout(root)
    hbox.setContentsMargins(6, 6, 6, 6)
    hbox.setSpacing(6)

    # 左側容器（工具栏 + 圖表）
    left_container = QtWidgets.QWidget()
    left_layout = QtWidgets.QVBoxLayout(left_container)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(2)

    # 工具栏
    toolbar = QtWidgets.QWidget()
    toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
    toolbar_layout.setContentsMargins(2, 2, 2, 2)
    toolbar_layout.setSpacing(5)

    btn_h_line = QtWidgets.QPushButton("H")
    btn_h_line.setFixedSize(40, 30)
    btn_h_line.setToolTip("畫水平線")
    
    btn_l_line = QtWidgets.QPushButton("L")
    btn_l_line.setFixedSize(40, 30)
    btn_l_line.setToolTip("畫普通線")

    btn_fibo = QtWidgets.QPushButton("Fibo")
    btn_fibo.setFixedSize(50, 30)
    btn_fibo.setToolTip("斐波那契水平線 (倍數: 0, 0.5, 1, 2, 3)")
    
    btn_text = QtWidgets.QPushButton("T")
    btn_text.setFixedSize(40, 30)
    btn_text.setToolTip("打字")

    btn_range = QtWidgets.QPushButton("📏")
    btn_range.setFixedSize(40, 30)
    btn_range.setToolTip("測量價格範圍")

    btn_screenshot = QtWidgets.QPushButton("Shot")
    btn_screenshot.setFixedSize(60, 30)
    btn_screenshot.setToolTip("截圖並匯出圖表")

    toolbar_layout.addWidget(btn_h_line)
    toolbar_layout.addWidget(btn_l_line)
    toolbar_layout.addWidget(btn_fibo)
    toolbar_layout.addWidget(btn_text)
    toolbar_layout.addWidget(btn_range)
    toolbar_layout.addWidget(btn_screenshot)
    toolbar_layout.addStretch()

    left_layout.addWidget(toolbar)

    win = pg.GraphicsLayoutWidget(title=f"{coin_name} Replay UI 5m")
    left_layout.addWidget(win)

    hbox.addWidget(left_container, 3)

    time_axis = TimeAxis(orientation="bottom")
    pg.setConfigOption('background', '#181c27')
    pg.setConfigOption('foreground', 'white')

    # ================================================================
    # ⭐ 資訊欄（上方顯示四欄，對齊Y位置）
    # ================================================================
    top_panel = pg.LabelItem(color="white", size="12pt")
    win.addItem(top_panel, row=0, col=0)

    top_panel.setMinimumHeight(90)
    top_panel.setMaximumHeight(130)

    # ================================================================
    # ⭐ 上方右側占位符（空白）
    # ================================================================
    spacer_top_right = pg.LabelItem(color="white", size="12pt")
    spacer_top_right.setText("")
    win.addItem(spacer_top_right, row=0, col=1)
    spacer_top_right.setMaximumHeight(130)

    # ================================================================
    # 圖表放在下方左側
    # ================================================================
    plot = win.addPlot(axisItems={"bottom": time_axis}, row=1, col=0)
    plot.getViewBox().setBackgroundColor("#181c27")

    plot.setAutoVisible(y=True)
    plot.showGrid(x=False, y=True, alpha=0.3)

    plot.showAxis("right", True)
    plot.showAxis("left", False)

    # ================================================================
    # ⭐ 右侧資訊欄（游標Y、NY_time，與Y軸並列）
    # ================================================================
    right_cursor_panel = pg.LabelItem(color="white", size="13pt")
    win.addItem(right_cursor_panel, row=1, col=1)
    
    right_cursor_panel.setMinimumWidth(200)
    right_cursor_panel.setMaximumWidth(240)

    # ================================================================
    # 十字游標線
    # ================================================================
    vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("gray", width=1))
    hline = pg.InfiniteLine(angle=0, pen=pg.mkPen("gray", width=1))
    plot.addItem(vline, ignoreBounds=True)
    plot.addItem(hline, ignoreBounds=True)

    # ================================================================
    # 按鈕功能連接
    # ================================================================
    def on_h_line_clicked():
        nonlocal line_mode
        line_mode = "horizontal"

    def on_l_line_clicked():
        nonlocal line_mode, line_start, temp_line
        line_mode = True
        line_start = None
        if temp_line:
            plot.removeItem(temp_line)
            temp_line = None

    def on_fibo_clicked():
        nonlocal line_mode, fibo_mode
        fibo_mode = True
        fibo_base_price = None

    def on_text_clicked():
        nonlocal text_mode
        text_mode = True

    def on_range_clicked():
        nonlocal range_mode, range_start_price
        range_mode = True
        range_start_price = None

    def on_screenshot_clicked():
        pixmap = win.grab()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "匯出截圖",
            "chart.png",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
        )
        if not file_path:
            return
        ok = pixmap.save(file_path)
        if ok:
            QtWidgets.QMessageBox.information(None, "截圖完成", f"已儲存:\n{file_path}")
        else:
            QtWidgets.QMessageBox.critical(None, "截圖失敗", "無法儲存檔案")

    btn_h_line.clicked.connect(on_h_line_clicked)
    btn_l_line.clicked.connect(on_l_line_clicked)
    btn_fibo.clicked.connect(on_fibo_clicked)
    btn_text.clicked.connect(on_text_clicked)
    btn_range.clicked.connect(on_range_clicked)
    btn_screenshot.clicked.connect(on_screenshot_clicked)

    # ================================================================
    # ⭐⭐ 模擬下單面板（多空、類型、價格、PnL）
    # 使用 LayoutWidget 內嵌標準 Qt 控制項
    # ================================================================
    order_panel = pg.LayoutWidget()
    order_panel.setMinimumWidth(260)
    hbox.addWidget(order_panel, 1)

    lbl_side = QtWidgets.QLabel("Side")
    side_combo = QtWidgets.QComboBox()
    side_combo.addItems(["long", "short"])

    lbl_type = QtWidgets.QLabel("Type")
    type_combo = QtWidgets.QComboBox()
    # 支援 market、limit、stop market
    type_combo.addItems(["market", "limit", "stop market"]) 

    lbl_price = QtWidgets.QLabel("Price")
    price_edit = QtWidgets.QLineEdit()
    price_edit.setPlaceholderText("e.g. 3500.0 (optional for market)")

    lbl_qty = QtWidgets.QLabel("Quantity")
    qty_spin = QtWidgets.QDoubleSpinBox()
    qty_spin.setDecimals(4)
    qty_spin.setMinimum(0.0001)
    qty_spin.setMaximum(1_000_000)
    qty_spin.setSingleStep(1.0)
    qty_spin.setValue(1.0)

    btn_place = QtWidgets.QPushButton("Place Order")
    btn_close = QtWidgets.QPushButton("Close Position")

    # 掛單表格與取消
    orders_table = QtWidgets.QTableWidget()
    orders_table.setColumnCount(6)
    orders_table.setHorizontalHeaderLabels(["ID", "Side", "Type", "Price", "Qty", "Status"])
    orders_table.horizontalHeader().setStretchLastSection(True)
    orders_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    orders_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    btn_cancel_selected = QtWidgets.QPushButton("Cancel Selected")

    # 成交明細表格
    lbl_trades = QtWidgets.QLabel("Trade History")
    trades_table = QtWidgets.QTableWidget()
    trades_table.setColumnCount(7)
    trades_table.setHorizontalHeaderLabels(["Time", "Side", "Price", "Qty", "PnL", "R_PnL", "Action"])
    trades_table.horizontalHeader().setStretchLastSection(True)
    trades_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    trades_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    lbl_unreal = QtWidgets.QLabel("Unrealized PnL: 0.00")
    lbl_real = QtWidgets.QLabel("Realized PnL: 0.00")
    lbl_pos = QtWidgets.QLabel("Position: flat")

    # 佈局
    order_panel.addWidget(lbl_side, row=0, col=0)
    order_panel.addWidget(side_combo, row=0, col=1)
    order_panel.addWidget(lbl_type, row=1, col=0)
    order_panel.addWidget(type_combo, row=1, col=1)
    order_panel.addWidget(lbl_price, row=2, col=0)
    order_panel.addWidget(price_edit, row=2, col=1)
    order_panel.addWidget(lbl_qty, row=3, col=0)
    order_panel.addWidget(qty_spin, row=3, col=1)
    order_panel.addWidget(btn_place, row=4, col=0, colspan=2)
    order_panel.addWidget(btn_close, row=5, col=0, colspan=2)
    order_panel.addWidget(lbl_unreal, row=6, col=0, colspan=2)
    order_panel.addWidget(lbl_real, row=7, col=0, colspan=2)
    order_panel.addWidget(lbl_pos, row=8, col=0, colspan=2)
    order_panel.addWidget(orders_table, row=9, col=0, colspan=2)
    order_panel.addWidget(btn_cancel_selected, row=10, col=0, colspan=2)
    order_panel.addWidget(lbl_trades, row=11, col=0, colspan=2)
    order_panel.addWidget(trades_table, row=12, col=0, colspan=2)
    
    btn_export_trades = QtWidgets.QPushButton("Export Trade History")
    order_panel.addWidget(btn_export_trades, row=13, col=0, colspan=2)

    # 狀態：部位 / 未成交委託 / 已實現 PnL / 成交紀錄
    position = None  # {"side": "long"|"short", "entry": float, "qty": float}
    realized_pnl = 0.0
    realized_r_pnl = 0.0
    pending_orders = []  # list of {id, side, type, price, qty, status}
    order_id_seq = 1
    trade_history = []  # list of {time, side, price, qty, pnl, r_pnl, action}
    trade_markers = []  # 成交標記（三角形）圖形物件

    def current_price():
        # 目前已繪製最後一根的收盤價作為成交基準
        if idx > 0:
            i = max(0, idx - 1)
            return float(df.iloc[i]["close"])
        return None

    def side_mult(side: str) -> int:
        return 1 if side == "long" else -1

    def update_pnl_labels():
        cp = current_price()
        unreal = 0.0
        unreal_r = 0.0
        if position is not None and cp is not None:
            unreal = (cp - position["entry"]) * side_mult(position["side"]) * position["qty"]
            unreal_r = unreal / max_SL
        lbl_unreal.setText(f"Unrealized PnL: {unreal:.3f} ({unreal_r:.3f}R)")
        lbl_real.setText(f"Realized PnL: {realized_pnl:.3f} ({realized_r_pnl:.3f}R)")
        if position is None:
            lbl_pos.setText("Position: flat")
        else:
            lbl_pos.setText(
                f"Position: {position['side']} {position['qty']:.4f} @ {position['entry']:.{price_precision_default}f}"
            )

    def refresh_orders_table():
        orders_table.setRowCount(len(pending_orders))
        for r, od in enumerate(pending_orders):
            orders_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(od['id'])))
            orders_table.setItem(r, 1, QtWidgets.QTableWidgetItem(od['side']))
            orders_table.setItem(r, 2, QtWidgets.QTableWidgetItem(od['type']))
            orders_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{od['price']:.{price_precision_default}f}"))
            orders_table.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{od['qty']:.4f}"))
            orders_table.setItem(r, 5, QtWidgets.QTableWidgetItem(od['status']))

    def refresh_trades_table():
        trades_table.setRowCount(len(trade_history))
        for r, trade in enumerate(trade_history):
            trades_table.setItem(r, 0, QtWidgets.QTableWidgetItem(trade['time']))
            trades_table.setItem(r, 1, QtWidgets.QTableWidgetItem(trade['side']))
            trades_table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{trade['price']:.{price_precision_default}f}"))
            trades_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{trade['qty']:.4f}"))
            pnl_text = f"{trade['pnl']:.3f}" if trade['pnl'] != 0 else "-"
            trades_table.setItem(r, 4, QtWidgets.QTableWidgetItem(pnl_text))
            r_pnl_text = f"{trade['r_pnl']:.3f}R" if trade['r_pnl'] != 0 else "-"
            trades_table.setItem(r, 5, QtWidgets.QTableWidgetItem(r_pnl_text))
            trades_table.setItem(r, 6, QtWidgets.QTableWidgetItem(trade['action']))

    def fill_order(fill_side: str, fill_price: float, fill_qty: float, fill_timestamp: float = 0, fill_time_str: str = ""):
        nonlocal position, realized_pnl, realized_r_pnl
        fprice = float(fill_price)
        fqty = float(fill_qty)
        if fqty <= 0:
            return
        
        current_time = fill_time_str
        current_ts = fill_timestamp
        
        if position is None:
            position = {"side": fill_side, "entry": fprice, "qty": fqty}
            # 記錄開倉
            trade_history.append({
                "time": current_time,
                "side": fill_side,
                "price": fprice,
                "qty": fqty,
                "pnl": 0.0,
                "r_pnl": 0.0,
                "action": "OPEN"
            })
            # 在圖表上添加三角形標記
            add_trade_marker(current_ts, fprice, fill_side, "OPEN")
        else:
            if position["side"] == fill_side:
                # 同向加倉：以加權平均價格更新
                new_qty = position["qty"] + fqty
                position["entry"] = (position["entry"] * position["qty"] + fprice * fqty) / new_qty
                position["qty"] = new_qty
                # 記錄加倉
                trade_history.append({
                    "time": current_time,
                    "side": fill_side,
                    "price": fprice,
                    "qty": fqty,
                    "pnl": 0.0,
                    "r_pnl": 0.0,
                    "action": "ADD"
                })
                # 在圖表上添加三角形標記
                add_trade_marker(current_ts, fprice, fill_side, "ADD")
            else:
                # 反向下單：先抵消
                close_qty = min(position["qty"], fqty)
                pnl = (fprice - position["entry"]) * side_mult(position["side"]) * close_qty
                r_pnl = pnl / max_SL
                realized_pnl += pnl
                realized_r_pnl += r_pnl
                # 記錄平倉
                trade_history.append({
                    "time": current_time,
                    "side": fill_side,
                    "price": fprice,
                    "qty": close_qty,
                    "pnl": pnl,
                    "r_pnl": r_pnl,
                    "action": "CLOSE"
                })
                # 在圖表上添加三角形標記
                add_trade_marker(current_ts, fprice, fill_side, "CLOSE")
                remaining = fqty - close_qty
                if position["qty"] > close_qty:
                    # 原部位剩餘
                    position["qty"] -= close_qty
                    if position["qty"] <= 0:
                        position = None
                else:
                    # 原部位全平
                    position = None
                if remaining > 0:
                    # 剩餘的以新方向開倉
                    position = {"side": fill_side, "entry": fprice, "qty": remaining}
                    trade_history.append({
                        "time": current_time,
                        "side": fill_side,
                        "price": fprice,
                        "qty": remaining,
                        "pnl": 0.0,
                        "r_pnl": 0.0,
                        "action": "OPEN"
                    })
                    # 在圖表上添加三角形標記（反手開倉）
                    add_trade_marker(current_ts, fprice, fill_side, "OPEN")
        refresh_trades_table()
        update_pnl_labels()

    def place_order():
        nonlocal order_id_seq
        s = side_combo.currentText()
        t = type_combo.currentText()
        p_text = price_edit.text().strip()
        q_val = float(qty_spin.value())

        cp = current_price()
        if cp is None:
            QtWidgets.QMessageBox.warning(None, "Order", "尚無K線，無法下單")
            return

        if t == "market":
            # 市價單立即以目前價成交
            fill_ts = 0
            fill_time = ""
            if idx > 0:
                fill_ts = df.iloc[idx-1]["dt_ny_ts"]
                fill_time = str(df.iloc[idx-1]["dt_ny"]).split('.')[0]
            fill_order(s, cp, q_val, fill_ts, fill_time)
            return

        # 非市價需要價格
        try:
            price_val = float(p_text)
        except Exception:
            QtWidgets.QMessageBox.warning(None, "Order", "請輸入有效價格")
            return

        order = {
            "id": order_id_seq,
            "side": s,
            "type": t,  # "limit" or "stop market"
            "price": price_val,
            "qty": q_val,
            "status": "open",
        }
        order_id_seq += 1
        pending_orders.append(order)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"已送出 {t} {s} @ {price_val}")
        refresh_orders_table()

    def close_position():
        nonlocal position, realized_pnl, realized_r_pnl
        if position is None:
            return
        cp = current_price()
        if cp is None:
            return
        pnl = (cp - position["entry"]) * side_mult(position["side"]) * position["qty"]
        r_pnl = pnl / max_SL
        realized_pnl += pnl
        realized_r_pnl += r_pnl
        
        # 記錄平倉
        fill_ts = 0
        fill_time = ""
        if idx > 0:
            fill_ts = df.iloc[idx-1]["dt_ny_ts"]
            fill_time = str(df.iloc[idx-1]["dt_ny"]).split('.')[0]
        close_side = "short" if position["side"] == "long" else "long"
        trade_history.append({
            "time": fill_time,
            "side": close_side,
            "price": cp,
            "qty": position["qty"],
            "pnl": pnl,
            "r_pnl": r_pnl,
            "action": "CLOSE"
        })
        # 在圖表上添加三角形標記
        add_trade_marker(fill_ts, cp, close_side, "CLOSE")
        
        position = None
        refresh_trades_table()
        update_pnl_labels()

    def cancel_selected_orders():
        selected = orders_table.selectionModel().selectedRows()
        if not selected:
            return
        ids = sorted([int(orders_table.item(r.row(), 0).text()) for r in selected], reverse=True)
        for oid in ids:
            for i, od in enumerate(list(pending_orders)):
                if od.get('id') == oid and od.get('status') == 'open':
                    pending_orders.pop(i)
                    break
        refresh_orders_table()

    def export_trade_history():
        if not trade_history:
            QtWidgets.QMessageBox.information(None, "Export", "No trade history to export")
            return
        
        # 使用檔案對話框選擇儲存位置
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, 
            "Export Trade History", 
            "trade_history.csv", 
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if not file_path:
            return
        
        try:
            # 轉換為 DataFrame
            df_trades = pd.DataFrame(trade_history)
            
            if file_path.endswith('.xlsx'):
                df_trades.to_excel(file_path, index=False)
            else:
                df_trades.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            QtWidgets.QMessageBox.information(None, "Export", f"Trade history exported to:\n{file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Export Error", f"Failed to export:\n{str(e)}")
    
    btn_place.clicked.connect(place_order)
    btn_close.clicked.connect(close_position)
    btn_cancel_selected.clicked.connect(cancel_selected_orders)
    btn_export_trades.clicked.connect(export_trade_history)

    # ================================================================
    # 成交標記功能
    # ================================================================
    def add_trade_marker(timestamp, price, side, action):
        """在圖表上添加成交三角形標記"""
        if timestamp == 0 or price == 0:
            return
        
        # long: 綠色上三角 ▲, short: 紅色下三角 ▼
        if side == "long":
            symbol = 't1'  # 上三角
            color = (0, 255, 0)  # 綠色
        else:
            symbol = 't'  # 下三角
            color = (255, 0, 0)  # 紅色
        
        scatter = pg.ScatterPlotItem(
            pos=[[timestamp, price]],
            size=8,
            symbol=symbol,
            brush=pg.mkBrush(color),
            pen=pg.mkPen('w', width=1)
        )
        plot.addItem(scatter)
        trade_markers.append(scatter)

    # MA 曲線
    ma_curve_20 = plot.plot(pen=pg.mkPen("green", width=2))
    ma_curve_100 = plot.plot(pen=pg.mkPen("orange", width=2))

    # ================================================================
    # 畫 K 線 function
    # ================================================================
    def draw_bar(record):
        t = record["dt_ny"].timestamp()
        o, h, l, c = record["open"], record["high"], record["low"], record["close"]

        wick_pen = pg.mkPen("white", width=1.3)


        plot.addItem(pg.PlotCurveItem(
            x=[t, t],
            y=[l, h],
            pen=wick_pen
        ))

        body_color = (31, 121, 245) if c >= o else (247, 82, 95)

        body_top = max(o, c)
        body_bottom = min(o, c)
        body_height = body_top - body_bottom
        center_y = (body_top + body_bottom) / 2
        w = 24*N_minutes

        body = pg.BarGraphItem(
            x=[t],
            width=2 * w,
            height=[body_height],
            y=[center_y],
            brush=body_color,
            pen=None
        )
        plot.addItem(body)

    # 檢查委託是否觸發成交（以本根K的最高/最低判斷觸發）
    def process_pending_orders(current_bar):
        if not pending_orders:
            return
        high_p = float(current_bar["high"]) if "high" in current_bar else None
        low_p = float(current_bar["low"]) if "low" in current_bar else None
        # 取得當前K線的時間戳和時間字串
        fill_ts = float(current_bar["dt_ny_ts"]) if "dt_ny_ts" in current_bar else 0
        fill_time = str(current_bar["dt_ny"]).split('.')[0] if "dt_ny" in current_bar else ""
        to_close = []
        for od in pending_orders:
            if od.get("status") != "open":
                continue
            s = od["side"]
            typ = od["type"]
            px = float(od["price"])
            qv = float(od["qty"])
            filled = False
            # 只有當委託價格介於該根K線的高低點之間（有碰到）才成交
            if low_p is not None and high_p is not None:
                touched = (low_p <= px <= high_p)
                if typ in ("limit", "stop market") and touched:
                    filled = True
            if filled:
                od["status"] = "filled"
                fill_order(s, px, qv, fill_ts, fill_time)
                to_close.append(od)
        # 清除已成交
        if to_close:
            for od in to_close:
                try:
                    pending_orders.remove(od)
                except ValueError:
                    pass
            refresh_orders_table()
    def draw_vertical_grids():
    # 先刪掉舊 grid
        for line in grid_lines:
            plot.removeItem(line)
        grid_lines.clear()

        # 根據 idx 畫新的 grid
        for i in range(idx):
            minute = df.iloc[i]["dt_ny"].minute
            if minute not in (0, 15, 30, 45):
                continue

            ts = df.iloc[i]["dt_ny_ts"]
            line = pg.InfiniteLine(
                ts, angle=90,
                pen=pg.mkPen((255, 255, 255, 40), width=1)
            )
            plot.addItem(line)
            grid_lines.append(line)

    # ================================================================
    # redraw（倒帶）
    # ================================================================
    def redraw_all():
        plot.clear()

        nonlocal ma_curve_20, ma_curve_100
        ma_curve_20 = plot.plot(pen=pg.mkPen("green", width=2))
        if N_minutes == 1:
            ma_curve_100 = plot.plot(pen=pg.mkPen("orange", width=2))

        for i in range(1, idx):
            bar = bars[i]
            draw_bar(bar)

            offset = (bar["high"] - bar["low"]) * 0.3
            txt = pg.TextItem(
                text=str(bar["bar_index"]),
                color="white",
                anchor=(0.5, 1.0)
            )
            txt.setPos(bar["dt_ny"].timestamp(), bar["low"] - offset)
            plot.addItem(txt)

        if idx > 1:
            ma_vals = df.iloc[:idx]["MA20"].values
            ma_vals_100 = df.iloc[:idx]["MA100"].values
            times = df.iloc[:idx]["dt_ny_ts"].values
            ma_curve_20.setData(times, ma_vals)
            if N_minutes == 1:
                ma_curve_100.setData(times, ma_vals_100)

        plot.addItem(vline, ignoreBounds=True)
        plot.addItem(hline, ignoreBounds=True)
        draw_vertical_grids()
        
        # 重繪成交標記
        for marker in trade_markers:
            plot.addItem(marker)
        
        # 重繪保存的劃線
        for line in saved_lines:
            plot.addItem(line)
        
        # 重繪後更新 PnL 顯示
        update_pnl_labels()

    # ================================================================
    # 滑鼠移動
    # ================================================================
    def mouseMoved(evt):
        pos = evt[0]
        if not plot.sceneBoundingRect().contains(pos):
            return

        mouse_point = plot.vb.mapSceneToView(pos)

        vline.setPos(mouse_point.x())
        hline.setPos(mouse_point.y())

        # 找最近 K
        mx = mouse_point.x()
        nearest_idx = (df["dt_ny_ts"] - mx).abs().idxmin()
        row = df.iloc[nearest_idx]
        
        # 游標Y位置（價格）
        cursor_y = mouse_point.y()
        ny_time = str(row['dt_ny']).split(' ')[1].split('-')[0]
        
        # 價格精度（依當日開盤價）
        price_prec = int(row.get("price_precision", price_precision_default))
        price_fmt = f"{{:.{price_prec}f}}"
        fmt_price = lambda val: price_fmt.format(val)

        # 右侧資訊欄（游標Y和NY_time）
        right_cursor_panel.setText(
            f"<span style='font-size:12pt;'>"
            f"<b>Cursor Y:</b><br>{fmt_price(cursor_y)}<br><br>"
            f"<b>NY_time:</b><br>{ny_time}"
            f"</span>"
        )
        
        # 上方資訊欄（四欄佈局，保留所有原本資訊）
        col1 = (
            f"<b>OHLC</b><br>-------------<br>"
            f"O: {fmt_price(row['open'])}<br>"
            f"H: {fmt_price(row['high'])}<br>"
            f"L: {fmt_price(row['low'])}<br>"
            f"C: {fmt_price(row['close'])}"
        )
        
        col2 = f"<b>SB info</b><br>-------------<br>ΔC: {row['delta_c']:.3f}%<br>size: {row['K_range']:.3f}% ({row['K_range_level']})<br>BBR: {row['body_ratio']:.3f} ({row['body_ratio_level']})<br>Shadow: {row['shadow_ratio']:.3f} <br>C Shadow: {row['counter_shadow_ratio']:.3f}"
        
        col3 = (
            f"<b>EB profit</b><br>-------------<br>"
            f"C2_profit_R: {row['C2_profit_R']:.3f}<br>"
            f"Max_K2_profit_R: {row['Max_K2_profit_R']:.3f}<br>"
            f"MA20: {fmt_price(row['MA20'])}"
        )
        if N_minutes == 1:
            col3 += f"<br>MA100: {fmt_price(row['MA100'])}"
        
        col4 = (
            f"<b>Entry info</b><br>-------------<br>"
            f"entry: {fmt_price(row['entry_price'])}<br>"
            f"SL: {fmt_price(row['SL_price'])}<br>"
            f"size: {row['size']:.3f}<br>"
        )
        col5 = (
            f"<b>R Levels</b><br>-------------<br>"
            f"0.4R: {fmt_price(row['0.4R'])}<br>"
            f"0.7R: {fmt_price(row['0.7R'])}<br>"
            f"1.0R: {fmt_price(row['1.0R'])}"
        )
        col6= (
            f"<b>R Levels</b><br>-------------<br>"
            f"1.3R: {fmt_price(row['1.3R'])}<br>"
            f"1.6R: {fmt_price(row['1.6R'])}<br>"
            f"2.0R: {fmt_price(row['2.0R'])}"
        )

        
        top_panel.setText(
            f"<span style='font-size:11pt;'>"
            f"<table width='100%' cellspacing='6' cellpadding='2'>"
            f"<tr>"
            f"<td style='vertical-align:top;'>{col3}</td>"
            f"<td style='vertical-align:top;'>{col1}</td>"
            f"<td style='vertical-align:top;'>{col5}</td>"
            f"<td style='vertical-align:top;'>{col6}</td>"
            f"<td style='vertical-align:top;'>{col2}</td>"
            f"<td style='vertical-align:top;'>{col4}</td>"
            f"</tr>"
            f"</table>"
            f"</span>"
        )

    proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

    # ================================================================
    # 自訂文字輸入對話框
    # ================================================================
    class TextInputDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("輸入文字")
            self.resize(400, 300)
            
            layout = QtWidgets.QVBoxLayout(self)
            
            # 文字輸入區
            label = QtWidgets.QLabel("請輸入文字 (Shift+Enter 換行):")
            layout.addWidget(label)
            
            self.text_edit = QtWidgets.QTextEdit()
            self.text_edit.setMinimumHeight(100)
            layout.addWidget(self.text_edit)
            
            # 顏色選擇
            color_layout = QtWidgets.QHBoxLayout()
            color_label = QtWidgets.QLabel("顏色:")
            color_layout.addWidget(color_label)
            
            self.color_buttons = []
            colors = [
                ("", (255, 255, 255)),
                ("", (255, 80, 80)),
                ("", (255, 160, 0)),
                ("", (255, 240, 100)),
                ("", (100, 200, 100)),
                ("", (100, 180, 255)),
                ("", (200, 120, 255))
            ]
            
            self.color_group = QtWidgets.QButtonGroup(self)
            for i, (name, rgb) in enumerate(colors):
                btn = QtWidgets.QPushButton(name)
                btn.setFixedSize(50, 30)
                btn.setCheckable(True)
                btn.color_rgb = rgb
                btn.color_name = name
                # 預設未選中樣式
                btn.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 2px solid gray;")
                btn.toggled.connect(lambda checked, b=btn: self.update_color_button_style(b, checked))
                self.color_group.addButton(btn, i)
                color_layout.addWidget(btn)
                self.color_buttons.append(btn)
            
            self.color_buttons[0].setChecked(True)  # 預設白色
            color_layout.addStretch()
            layout.addLayout(color_layout)
            
            # 字體大小選擇
            size_layout = QtWidgets.QHBoxLayout()
            size_label = QtWidgets.QLabel("字體大小:")
            size_layout.addWidget(size_label)
            
            self.size_combo = QtWidgets.QComboBox()
            self.size_combo.addItems(["8", "10", "12", "14", "16", "18", "20", "24", "28", "32"])
            self.size_combo.setCurrentText("12")
            size_layout.addWidget(self.size_combo)
            size_layout.addStretch()
            layout.addLayout(size_layout)
            
            # 按鈕
            button_layout = QtWidgets.QHBoxLayout()
            ok_btn = QtWidgets.QPushButton("確定")
            cancel_btn = QtWidgets.QPushButton("取消")
            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)
            button_layout.addStretch()
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
        
        def update_color_button_style(self, button, checked):
            """更新顏色按鈕樣式，選中時顯示粗外框"""
            rgb = button.color_rgb
            if checked:
                # 選中：粗黑框 + 陰影
                button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 3px solid black; font-weight: bold;")
            else:
                # 未選中：細灰框
                button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 2px solid gray;")
        
        def get_values(self):
            text = self.text_edit.toPlainText()
            checked_btn = self.color_group.checkedButton()
            color = checked_btn.color_rgb if checked_btn else (255, 255, 255)
            size = int(self.size_combo.currentText())
            return text, color, size

    # ================================================================
    # TradingView 風格色票
    # ================================================================
    def get_tradingview_color():
        """顯示 TradingView 風格的顏色選擇器"""
        dialog = QtWidgets.QColorDialog()
        
        # TradingView 基本色票（對應 Basic colors 區域，6行8列 = 48色）
        tv_basic_colors = [
            # Row 1 (was Col 1)
            (242, 54, 69), (252, 203, 205), (250, 161, 164), (247, 124, 128), (247, 82, 95),(242, 54, 69),
            # Row 2 (was Col 2)
            (255, 152, 0), (255, 224, 178), (255, 204, 128), (255, 183, 77), (255, 167, 38),(255, 152, 0),
            # Row 3 (was Col 3)
            (255, 235, 59), (255, 249, 196), (255, 245, 157), (255, 241, 118), (255, 238, 88),(255, 235, 59),
            # Row 4 (was Col 4)
            (76, 175, 80), (200, 230, 201), (165, 214, 167), (129, 199, 132), (102, 187, 106),(76, 175, 80),
            # Row 5 (was Col 5)
            (8, 153, 129), (172, 229, 220), (112, 204, 189), (66, 189, 168), (34, 171, 148),(8, 153, 129),
            # Row 6 (was Col 6)
            (0, 188, 212), (178, 235, 242), (128, 222, 234), (77, 208, 225), (38, 198, 218),(0, 188, 212),
            # Row 7 (was Col 7)
            (41, 98, 255), (187, 217, 251), (144, 191, 249), (91, 156, 246), (49, 121, 245), (41, 98, 255),
            # Row 8 (was Col 8)
            (103, 58, 183), (209, 196, 233), (179, 157, 219), (149, 117, 205), (126, 87, 194), (103, 58, 183)
        ]
        
        # 設定基本顏色（Basic colors 區域）
        for i, (r, g, b) in enumerate(tv_basic_colors):
            if i < 48:  # Qt 的 standard colors 有 48 個
                dialog.setStandardColor(i, QtGui.QColor(r, g, b))
        
        if dialog.exec():
            return dialog.currentColor()
        return None

    # ================================================================
    def mouseClicked(evt):
        nonlocal line_mode, line_start, temp_line, selected_item, fibo_mode, fibo_base_price, text_mode, fibo_groups, line_custom_pens, range_mode, range_start_price, price_ranges

        pos = evt.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return

        point = plot.vb.mapSceneToView(pos)
        x, y = point.x(), point.y()

        # 檢查是否點擊了文字或線（若在 fibo/text/range 模式，略過現有線的選取以便建立新物件）
        clicked_item = None
        clicked_type = None  # "text" 或 "line"

        ignore_hit_test = fibo_mode or text_mode or range_mode

        # 先檢查文字，使用 boundingRect
        if not ignore_hit_test:
            for item in saved_lines:
                if not item.isVisible():
                    continue
                if isinstance(item, pg.TextItem):
                    local_pos = item.mapFromScene(pos)
                    if item.boundingRect().contains(local_pos):
                        clicked_item = item
                        clicked_type = "text"
                        break

        # 再檢查劃線
        if not ignore_hit_test and clicked_item is None:
            for item in saved_lines:
                if not item.isVisible() or isinstance(item, pg.TextItem):
                    continue
                try:
                    line_start_pos = item.getState()['points'][0]
                    line_end_pos = item.getState()['points'][1]

                    x1, y1 = line_start_pos[0], line_start_pos[1]
                    x2, y2 = line_end_pos[0], line_end_pos[1]

                    if abs(x2 - x1) < 0.1:
                        dist = abs(x - x1)
                    elif abs(y2 - y1) < 0.1:
                        dist = abs(y - y1)
                    else:
                        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
                        den = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
                        dist = num / den if den != 0 else float('inf')

                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    if x_min - 10 <= x <= x_max + 10 and y_min - 10 <= y <= y_max + 10 and dist <= 10:
                        clicked_item = item
                        clicked_type = "line"
                        break
                except Exception:
                    continue

        # 右鍵：文字或線的上下文菜單
        if evt.button() == QtCore.Qt.RightButton:
            if clicked_item:
                if clicked_type == "text":
                    target_text = clicked_item
                    menu = QtWidgets.QMenu()
                    del_act = menu.addAction("刪除")
                    color_act = menu.addAction("改顏色")
                    size_act = menu.addAction("改大小")

                    def delete_text():
                        nonlocal selected_item
                        try:
                            plot.removeItem(target_text)
                        except Exception:
                            pass
                        if target_text in saved_lines:
                            saved_lines.remove(target_text)
                        if selected_item == target_text:
                            selected_item = None

                    def change_text_color():
                        color = get_tradingview_color()
                        if color and color.isValid():
                            target_text.setColor(color)

                    def change_text_size():
                        font = QtGui.QFont()
                        if hasattr(target_text, "textItem"):
                            font = target_text.textItem.font()
                        current_size = font.pointSize() if font.pointSize() > 0 else 12
                        new_size, ok = QtWidgets.QInputDialog.getInt(None, "字體大小", "字體大小 (pt):", current_size, 6, 64)
                        if ok:
                            font.setPointSize(int(new_size))
                            try:
                                target_text.textItem.setFont(font)
                            except Exception:
                                target_text.setFont(font)

                    del_act.triggered.connect(delete_text)
                    color_act.triggered.connect(change_text_color)
                    size_act.triggered.connect(change_text_size)
                    menu.exec(QtGui.QCursor.pos())
                else:
                    target_line = clicked_item

                    target_group = None
                    for g in fibo_groups:
                        if target_line in g.get("lines", []):
                            target_group = g
                            break

                    context_menu = QtWidgets.QMenu()
                    delete_action = context_menu.addAction("刪除")
                    color_action = context_menu.addAction("改顏色")

                    def delete_line():
                        nonlocal selected_item
                        # 检查是否为价格范围测量线
                        target_range = None
                        if isinstance(target_line, pg.LineSegmentROI):
                            for rg in price_ranges:
                                if rg.get('line') == target_line:
                                    target_range = rg
                                    break
                        
                        if target_range:
                            # 删除测量线和标签
                            try:
                                plot.removeItem(target_range['line'])
                            except Exception:
                                pass
                            try:
                                plot.removeItem(target_range['label'])
                            except Exception:
                                pass
                            
                            # 从saved_lines中移除
                            if target_range['line'] in saved_lines:
                                saved_lines.remove(target_range['line'])
                            if target_range['label'] in saved_lines:
                                saved_lines.remove(target_range['label'])
                            
                            price_ranges.remove(target_range)
                            if selected_item == target_range['line']:
                                selected_item = None
                        elif target_group:
                            for ln in list(target_group["lines"]):
                                if ln in saved_lines:
                                    try:
                                        plot.removeItem(ln)
                                    except Exception:
                                        pass
                                    saved_lines.remove(ln)
                                    if ln in line_custom_pens:
                                        del line_custom_pens[ln]
                            fibo_groups.remove(target_group)
                            if selected_item in target_group["lines"]:
                                selected_item = None
                        else:
                            try:
                                plot.removeItem(target_line)
                            except Exception:
                                pass
                            if target_line in saved_lines:
                                saved_lines.remove(target_line)
                            if target_line in line_custom_pens:
                                del line_custom_pens[target_line]
                            if selected_item == target_line:
                                selected_item = None

                    def change_color():
                        nonlocal line_custom_pens
                        color = get_tradingview_color()
                        if color and color.isValid():
                            r, g, b = color.red(), color.green(), color.blue()
                            pen = pg.mkPen((r, g, b), width=2)
                            if target_group:
                                for ln in target_group["lines"]:
                                    ln.setPen(pen)
                                    line_custom_pens[ln] = pen
                            else:
                                target_line.setPen(pen)
                                line_custom_pens[target_line] = pen

                    delete_action.triggered.connect(delete_line)
                    color_action.triggered.connect(change_color)
                    context_menu.exec(QtGui.QCursor.pos())
            return

        # 左鍵點擊邏輯
        if evt.button() != QtCore.Qt.LeftButton:
            return

        if clicked_item:
            if clicked_type == "line":
                # 切換選中狀態並高亮線條
                if isinstance(selected_item, pg.LineSegmentROI) and selected_item is not clicked_item:
                    # 恢復上一個選中線的原始顏色
                    original_pen = line_custom_pens.get(selected_item, pg.mkPen("cyan", width=2))
                    selected_item.setPen(original_pen)
                if selected_item == clicked_item:
                    # 取消選中，恢復原始顏色
                    original_pen = line_custom_pens.get(selected_item, pg.mkPen("cyan", width=2))
                    selected_item.setPen(original_pen)
                    selected_item = None
                else:
                    selected_item = clicked_item
                    clicked_item.setPen(pg.mkPen("yellow", width=3))
            else:
                # 選中文字，保留現有高亮
                if isinstance(selected_item, pg.LineSegmentROI):
                    original_pen = line_custom_pens.get(selected_item, pg.mkPen("cyan", width=2))
                    selected_item.setPen(original_pen)
                selected_item = clicked_item
            return

        if not line_mode and not fibo_mode and not text_mode and not range_mode:
            return
        
        # 處理價格範圍測量模式
        if range_mode:
            if range_start_price is None:
                # 第一次點擊：記錄起點
                range_start_price = y
                return
            else:
                # 第二次點擊：創建一條可調整端點的測量線
                y1 = range_start_price
                y2 = y
                
                # 確保 y1 < y2
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # 使用點擊位置的X座標
                x_pos = x
                
                # 創建一條垂直線，端點可拖動
                measure_line = pg.LineSegmentROI(
                    [(x_pos, y1), (x_pos, y2)],
                    pen=pg.mkPen((100, 200, 255), width=2)
                )
                measure_line.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.RightButton)
                
                plot.addItem(measure_line)
                
                # 計算價格差異
                price_diff = y2 - y1
                price_pct = (price_diff / y1) * 100 if y1 != 0 else 0
                
                # 創建文字標籤顯示價格資訊（放在線的右側中間）
                label_text = f"{price_diff:.3f} ({price_pct:.3f}%)"
                text_label = pg.TextItem(text=label_text, color=(100, 200, 255), anchor=(0, 0.5))
                x_range = plot.viewRange()[0]
                x_width = x_range[1] - x_range[0]
                text_label.setPos(x_pos + x_width * 0.02, (y1 + y2) / 2)
                plot.addItem(text_label)
                
                # 保存範圍物件（作為群組）
                range_group = {
                    'line': measure_line,
                    'label': text_label
                }
                price_ranges.append(range_group)
                saved_lines.append(measure_line)
                saved_lines.append(text_label)
                
                # 添加拖曳時更新標籤的回調
                def update_range_display():
                    # 獲取線的當前端點位置
                    points = measure_line.getState()['points']
                    
                    y_bottom = points[0][1]
                    y_top = points[1][1]
                    x_line = points[0][0]
                    
                    # 確保 y_bottom < y_top
                    if y_bottom > y_top:
                        y_bottom, y_top = y_top, y_bottom
                    
                    # 更新價格差異
                    new_diff = abs(y_top - y_bottom)
                    new_pct = (new_diff / y_bottom) * 100 if y_bottom != 0 else 0
                    text_label.setText(f"{new_diff:.3f} ({new_pct:.3f}%)")
                    
                    # 更新標籤位置（放在線的右側中間）
                    x_range = plot.viewRange()[0]
                    x_width = x_range[1] - x_range[0]
                    label_x = x_line + x_width * 0.02
                    text_label.setPos(label_x, (y_bottom + y_top) / 2)
                
                measure_line.sigRegionChanged.connect(update_range_display)
                
                range_mode = False
                range_start_price = None
                return
        
        # 處理 Fibo 模式（群組，同步調整）
        if fibo_mode:
            if fibo_base_price is None:
                # 第一次點擊：記錄基準價格
                fibo_base_price = y
                return
            else:
                # 第二次點擊：繪製斐波那契水平線
                fibo_multipliers = [0, 0.5, 1, 2, 3]
                colors = [(255, 255, 255), (255, 165, 0), (255, 255, 100), (100, 180, 255), (255, 150, 200)]  # 白、橘、黃、淺藍、粉
                
                # 只向右延伸：起點在點擊位置，終點為目前視圖右緣
                x_range = plot.viewRange()[0]
                x1 = x  # 點擊位置
                x2 = x_range[1]  # 視圖右邊界
                
                lines = []
                fibo_ys = []
                for idx_mult, (mult, color) in enumerate(zip(fibo_multipliers, colors)):
                    fibo_y = fibo_base_price + (y - fibo_base_price) * mult
                    line_pen = pg.mkPen(color, width=2)
                    line_roi = pg.LineSegmentROI([(x1, fibo_y), (x2, fibo_y)], pen=line_pen)
                    plot.addItem(line_roi)
                    # 第一條作為主線，其他線禁止滑鼠交互，避免單獨移動
                    if idx_mult == 0:
                        line_roi.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
                    else:
                        line_roi.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                    lines.append(line_roi)
                    fibo_ys.append(fibo_y)
                    saved_lines.append(line_roi)
                    line_custom_pens[line_roi] = line_pen  # 記錄初始顏色

                master_line = lines[0]

                group = {
                    "lines": lines,
                    "ys": fibo_ys,
                    "master": master_line,
                    "x_anchor": x1,
                    "last_points": master_line.getState()["points"],
                }
                fibo_groups.append(group)

                def sync_fibo_group():
                    anchor_x = group["x_anchor"]
                    p1, p2 = master_line.getState()["points"]
                    last_p1, last_p2 = group.get("last_points", [p1, p2])

                    dx = p1[0] - last_p1[0]
                    dy = p1[1] - last_p1[1]
                    # 更新錨點與 Y 座標（整組一起平移）
                    anchor_x += dx
                    group["x_anchor"] = anchor_x
                    group["ys"] = [yy + dy for yy in group["ys"]]

                    # 保持長度，僅向右延伸
                    new_len = max(p2[0] - p1[0], 1e-6)
                    new_x1 = anchor_x
                    new_x2 = anchor_x + new_len

                    for line_obj, base_y in zip(group["lines"], group["ys"]):
                        line_obj.blockSignals(True)
                        try:
                            line_obj.setPoints([new_x1, base_y], [new_x2, base_y])
                        except Exception:
                            pass
                        line_obj.blockSignals(False)

                    group["last_points"] = [p1, p2]

                master_line.sigRegionChanged.connect(sync_fibo_group)

                fibo_mode = False
                fibo_base_price = None
                return
        
        # 處理 Text 模式
        if text_mode:
            dialog = TextInputDialog()
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                text, color_rgb, font_size = dialog.get_values()
                if text:
                    # 在點擊位置添加文字
                    text_item = pg.TextItem(text=text, color=color_rgb, anchor=(0, 0))
                    text_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
                    # 左鍵拖曳、右鍵穿透到場景以觸發自訂選單
                    text_item.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
                    text_item.setZValue(5)
                    
                    # 設定字體大小
                    font = QtGui.QFont()
                    font.setPointSize(font_size)
                    try:
                        text_item.textItem.setFont(font)
                    except Exception:
                        pass
                    
                    text_item.setPos(x, y)
                    plot.addItem(text_item)
                    saved_lines.append(text_item)  # 保存文字項目
            text_mode = False
            return
        
        if not line_mode:
            return
        
        # 處理劃線模式
        if line_mode == "horizontal":
            # 水平線：使用當前視圖寬度的中間部分
            x_range = plot.viewRange()[0]
            x_width = x_range[1] - x_range[0]
            x1 = x_range[0] + x_width * 0.2  # 從左邊 20% 開始
            x2 = x_range[0] + x_width * 0.8  # 到右邊 80% 結束
            line_roi = pg.LineSegmentROI([(x1, y), (x2, y)], pen=pg.mkPen("cyan", width=2))
            line_roi.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
            plot.addItem(line_roi)
            saved_lines.append(line_roi)
            line_custom_pens[line_roi] = pg.mkPen("cyan", width=2)  # 記錄初始顏色
            line_mode = False
            return
        
        # 處理普通劃線模式
        if line_start is None:
            line_start = (x, y)
            return

        x1, y1 = line_start
        # 使用 LineSegmentROI 支持拖動端點和整條線移動
        line_roi = pg.LineSegmentROI([(x1, y1), (x, y)], pen=pg.mkPen("cyan", width=2))
        line_roi.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
        plot.addItem(line_roi)
        saved_lines.append(line_roi)  # 保存劃線
        line_custom_pens[line_roi] = pg.mkPen("cyan", width=2)  # 記錄初始顏色

        if temp_line:
            plot.removeItem(temp_line)
            temp_line = None

        line_start = None
        line_mode = False

    plot.scene().sigMouseClicked.connect(mouseClicked)

    # ================================================================
    # update（下一根）
    # ================================================================
    def update():
        nonlocal idx
        if idx >= total:
            return

        bar = bars[idx]
        draw_bar(bar)

        # 新K出現後，處理委託
        process_pending_orders(bar)

        ma_vals = df.iloc[:idx+1]["MA20"].values
        ma_vals_100 = df.iloc[:idx+1]["MA100"].values
        times = df.iloc[:idx+1]["dt_ny_ts"].values
        ma_curve_20.setData(times, ma_vals)
        if N_minutes == 1:
            ma_curve_100.setData(times, ma_vals_100)

        offset = (bar["high"] - bar["low"]) * 0.3
        txt = pg.TextItem(
            text=str(bar["bar_index"]),
            color="white",
            anchor=(0.5, 1.0)
        )
        txt.setPos(bar["dt_ny"].timestamp(), bar["low"] - offset)
        plot.addItem(txt)
        draw_vertical_grids()

        idx += 1
        update_pnl_labels()

    # ================================================================
    # key press
    # ================================================================
    def keyPress(evt):
        nonlocal idx, line_mode, line_start, temp_line, selected_item, fibo_groups, line_custom_pens
        key = evt.key()

        if key == QtCore.Qt.Key_Delete:
            if selected_item is not None:
                target_group = None
                if isinstance(selected_item, pg.LineSegmentROI):
                    for g in fibo_groups:
                        if selected_item in g.get("lines", []):
                            target_group = g
                            break
                if target_group:
                    for ln in list(target_group["lines"]):
                        if ln in saved_lines:
                            try:
                                plot.removeItem(ln)
                            except Exception:
                                pass
                            saved_lines.remove(ln)
                            if ln in line_custom_pens:
                                del line_custom_pens[ln]
                    fibo_groups.remove(target_group)
                else:
                    try:
                        plot.removeItem(selected_item)
                    except Exception:
                        pass
                    if selected_item in saved_lines:
                        saved_lines.remove(selected_item)
                    if selected_item in line_custom_pens:
                        del line_custom_pens[selected_item]
                selected_item = None
            return

        if key == QtCore.Qt.Key_Right:
            update()

        elif key == QtCore.Qt.Key_Left:
            if idx > 1:
                idx -= 1
                redraw_all()

    # 將鍵盤事件綁定在頂層視窗，確保可接收快捷鍵
    root.keyPressEvent = keyPress

    def on_close(event):
        QtWidgets.QApplication.quit()

    root.closeEvent = on_close

    root.resize(1400, 800)
    root.show()
    app.exec()


# ================================================================
# 程式進入點
# ================================================================
if __name__ == "__main__":
    main()
