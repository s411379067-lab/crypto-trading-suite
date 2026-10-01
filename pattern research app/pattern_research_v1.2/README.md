#### Pattern Research v1.2

這是依照新版 architecture 建立的第一個可迭代版本。UI 視覺沿用舊 `backtest_UI.py` 的 PyQtGraph 深色風格，但核心已拆成 `shared_core`、`pattern_analyzer`、`case_generator`。


##### v1.2 Text / Rectangle resize 修正

- Text 實體框線與選取框都改為完整四邊閉合線。
- Text 關閉自動換行時，套用設定會依最長一行自動 fit 寬度。
- Text / Rectangle 完全移除舊右上角縮放 handle，只保留右側中點與下側中點。
- Text 設定中的 B / I 按鈕啟用時會反白。

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

### v0.1.2 Fibo / palette / crosshair / X-axis update

- Added **Fibo drawing** using the legacy interaction model but a new JSON-persisted domain model:
  - Two-click start/end creation.
  - Default levels: 0 / 0.5 / 1 / 2 / 3.
  - 0x / 1x control handles.
  - Group box movement / horizontal extent adjustment.
  - Right-click: Fibo Settings / Delete / Change Color.
  - Level multipliers and per-level colors persist in Case JSON.
- All drawing color pickers now use the same fixed TradingView-style palette from the legacy program.
- New horizontal/trend/Fibo lines default to **white**.
- Added TradingView-like crosshair coordinate labels:
  - X snaps to the nearest revealed candle and shows date/time in the selected view timezone.
  - Y shows the current price on the right axis.
- Added independent **X-axis tick interval** control: `5m / 15m / 30m / 1H / 4H / 1D`.
  - Tick positions are aligned to exact local-time boundaries.
  - The choice is persisted as `display.x_tick_interval` in Case JSON.
  - This setting does not change chart timeframe or Replay step.

### v0.1.3 Rectangle / Order update

- Added **Rectangle drawing**:
  - Two-click creation.
  - Move and resize after creation.
  - Right-click `長方形設定` to change border color, fill color, fill opacity, and border width.
  - Rectangle geometry/style persist in `drawings[]` inside Case JSON.
  - Border defaults to white; all color pickers continue using the fixed legacy TradingView palette.
- Left side is now split vertically:
  - Top: **Order** module.
  - Bottom: **Case Library** / folder selection.
  - The splitter can be resized by dragging.
- Restored legacy-style Order simulation:
  - market / limit / stop market.
  - quantity, 1/3 P, 1/2 P, full P.
  - Place Order / Close Position.
  - Pending Orders and Cancel Selected.
  - Position, realized/unrealized PnL and R-PnL (`R=60`, matching legacy prototype).
  - Long/Short triangle fill markers on the chart.
  - Legacy S/L buttons beside the crosshair price to prefill side/type/price.
  - Export order records to CSV and Clean Records.
- Every Place Order / Close Position action is stored as one entry in Case JSON `orders[]`.
- A single order record can be deleted with `Delete Selected Record`; position/PnL are recomputed from the remaining filled order records.
- Cancelling a pending order preserves the order record and changes its status to `cancelled` rather than silently deleting it.

Example order record:

```json
{
  "id": "order-...",
  "created_ts": 0,
  "replay_time": "...",
  "side": "long",
  "order_type": "limit",
  "requested_price": 25000.0,
  "qty": 1.0,
  "status": "filled",
  "fill_ts": 0,
  "fill_price": 25000.0,
  "computed_action": "OPEN",
  "realized_pnl": 0.0,
  "realized_r_pnl": 0.0
}
```

### v0.1.4 Fibo control / filled-only order records

- Fibonacci 0/1 controls now use a **large invisible drag hit-zone** (~16 screen pixels high) while the visible Fibo lines stay thin. This makes vertical adjustment reliable across zoom levels.
- Fibo control hit-zones are recalculated whenever the chart range changes, so zooming no longer makes them too thin to grab.
- Pending limit / stop-market orders are now **session-only** and are not written to Case JSON.
- `orders[]` now stores **filled orders only**. A limit/stop order is appended to JSON only at the moment it fills.
- Cancelling an unfilled pending order simply removes it from the in-memory pending list and creates no history record.
- Existing v0.1.3 Case files are cleaned on load so open/cancelled order rows are removed from `orders[]`; filled records are retained.

### v0.1.5 Order table readability

- Pending Orders / Order Records table headers use a dark high-contrast style with bold light text.

### v0.1.6 Drawing Template Library

- Added a filesystem-backed `drawing template/` library. Templates are stored outside Case JSON and contain appearance/configuration only, never coordinates.
- Template categories:
  - `drawing template/line/` — shared by horizontal lines and trend lines.
  - `drawing template/rectangle/`
  - `drawing template/fibonacci/`
  - `drawing template/text/`
- Every supported drawing right-click menu now includes `模板 >` and only shows templates compatible with that drawing type.
- Drawing settings dialogs now include `存為模板...` and allow a custom template name.
- Reusing an existing template name asks before overwriting.
- Applying a template changes style/configuration only; geometry, price/time coordinates, and text content remain unchanged.
- Fibo templates store both levels (multiplier + color) and Fibo style.
- Right-click `刪除` is now always the final menu action for every drawing type.

### v0.1.8
- 以 v0.1.7 為基礎加入 **Ctrl Magnet**。
- 按住 Ctrl 時，十字游標會吸附到最近已揭露 K 棒的 O/H/L/C。
- 建立 Drawing、Line 端點 resize / 整體拖曳、Horizontal Line、Rectangle 拖曳/縮放、Fibo 控制與 Text 拖曳均接入磁鐵約束。
- Magnet 只使用 Replay 已揭露資料，不讀取未來 K 棒。


### v0.1.9
- 修正 Text Drawing 右鍵沒有選單的問題。
- Text 右鍵選單現在包含：文字設定、改顏色、改大小、模板，以及最底部的刪除。
- Text 內部文字物件不再攔截右鍵；拖曳與 Ctrl Magnet 功能維持不變。

### v0.1.10 Text Box Drawing
- Text 升級為可調整寬高的 Text Box Drawing。
- 文字設定改為 TradingView-like 編輯器：文字內容、文字顏色、字體大小、粗體、斜體、背景、框線、自動換行。
- 文字顏色與框線顏色完全獨立。
- 框線可以關閉；即使無框線，左鍵雙擊選取後仍顯示黃色選取框與 resize handles。
- 選取後可拖曳整個文字框，亦可拖曳 handles 修改寬、高；幾何資訊保存到 Case JSON 的 text `box`。
- Text Box 保留 Ctrl magnet，拖曳與 resize 時可吸附已揭露 K 棒。
- 舊版 text drawing 沒有 `box` / 新 style 欄位時會自動套用相容預設值。


### v1.0 — 第一版完整版

- 將 v0.1.10 之後的功能集合定義為第一版完整版。
- 新增 Drawing/Text **Ctrl+C / Ctrl+V**：
  - 必須先用左鍵雙擊選取 Drawing。
  - Ctrl+C 複製 Drawing Domain 資料，不複製 PyQtGraph 物件。
  - Ctrl+V 產生新的 Drawing ID，保留原本樣式、文字內容、Fibo levels、Text Box 寬高等設定。
  - 貼上的物件依目前畫面比例向右下偏移約 12px；連續貼上會逐次增加偏移，避免完全重疊。
  - 貼上後新 Drawing 自動成為目前選取物件並立即寫入 Case JSON。
  - Horizontal Line、Trend Line、Rectangle、Fibonacci、Text Box 全部支援。
- Drawing Clipboard 是 Pattern Analyzer 內部 clipboard，並刻意使用 Chart-local shortcut；在 Notes / Pattern / Order 文字欄位內的 Ctrl+C / Ctrl+V 仍維持正常文字剪貼功能。
- Clipboard 不會因切換 Case 清空，因此可複製 Drawing 後切換 Case 再貼上。

### v1.1 Text / Rectangle resize update

- New Text drawings start with a predictable default box size of about 360x180 screen pixels at creation time.
- If the initial text content needs more vertical space, the initial box height expands to contain it.
- Text resize handles changed from the top-right corner to two independent handles:
  - right-edge midpoint: adjusts width only;
  - bottom-edge midpoint: adjusts height only while keeping the top edge fixed.
- Rectangle resize handles use the same right-midpoint / bottom-midpoint interaction.
- Existing Text / Rectangle JSON geometry remains compatible; no schema migration is required.
