#### Pattern Research v0.1

這是依照新版 architecture 建立的第一個可迭代版本。UI 視覺沿用舊 `backtest_UI.py` 的 PyQtGraph 深色風格，但核心已拆成 `shared_core`、`pattern_analyzer`、`case_generator`。

##### v0.1 已包含

- 獨立 Case Generator：批次產生 A/B/C Case JSON。
- Folder-based Case Library。
- TXT / CSV / Parquet Market Data Adapter。
- Canonical UTC Market Data。
- Time-based Replay Engine；左右鍵以 1 分鐘前進/後退。
- View timeframe 與 Replay step 解耦。
- M1 / M5 / M15 / H1，使用 M1 raw data 即時聚合 Partial Candle。
- 任意 IANA timezone 顯示。
- 舊版風格 OHLC panel、Crosshair、Auto Scale、Screenshot。
- Drawing：水平線、趨勢線、文字；選取後可用 Delete 刪除。
- Drawing 寫入 Case JSON，重新開啟後還原；水平線/趨勢線可移動，文字位置會在 Auto Save 時同步。
- Pattern 多筆自由文字。
- 盤中紀錄，保存當下 Replay time。
- 1 秒 Debounced/Periodic Auto Save + atomic replace。

##### v0.1 暫不接入

- Fibo：舊版 UI / Manager 可在下一輪搬入 Drawing Adapter。
- Order / Position / PnL：舊版保留作為 prototype，下一輪接 Replay M1 market event，不刪除原始碼。
- SBS / MA Indicator：之後以 Indicator module 接入。
- Drawing 右鍵完整樣式設定（刪除已可使用 Delete）。
- Pattern Search / Statistics / Template。

##### 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

##### 使用順序

先啟動 Case Generator：

```bash
python run_case_generator.py
```

選擇 M1 歷史資料，設定 Symbol、日期與 A/B/C，產生 Case JSON。

再啟動 Pattern Analyzer：

```bash
python run_analyzer.py
```

左側選擇 Case Library 資料夾，雙擊 JSON 開啟。

##### Market Data 欄位

v0.1 支援至少：

```text
時間欄：dt_utc / timestamp / time_utc8 / time_utc3 / datetime / time
OHLC：open / high / low / close
可選：tick_volume / real_volume / volume / spread
```

時間欄若含 `+08:00`、`+03:00` 等 offset，載入後統一轉成 UTC timestamp。

##### 重要限制

Replay Step = 1 minute 時，原始資料必須至少是 M1。若輸入 M5 raw data，程式無法真實還原 M5 內部的 1 分鐘走法。

### v0.1.1 Drawing interaction update

- Added **AutoAll**: fits all candles currently revealed by Replay and continues following new Replay bars until the user manually pans/zooms.
- Drawing selection is now **left-button double-click only**. A single click no longer selects a drawing.
- Right-click drawing context menus now match the legacy UI behavior for the drawing types currently implemented:
  - Lines: Line Settings (color / solid-dashed-dotted / width), Delete, Change Color.
  - Text: Delete, Change Color, Change Size.
- Drawing style changes remain persisted in Case JSON.
