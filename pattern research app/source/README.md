# Pattern Research v2.11.1

## v2.11.1 — Analyzer metrics and responsive panels

- Trade Metrics now show replay-day win rate as a percentage with `wins/all` counts; only completed round-trip trades are included.
- The Analyzer chart toolbar uses multiple rows, and the resizable side panels have explicit minimum widths and no longer collapse while dragging.
- The Analyzer window minimum width is reduced to fit standard desktop displays while preserving usable chart and panel widths.


## Version history

## v2.9.0 - Fixed source folder and repository data separation

- The active application source now lives in the fixed `source/` folder.
- Version history is managed by Git commits, tags, and GitHub Releases instead of versioned source folders.
- Local Case JSON files and NAS100 market-data files are excluded from future Git commits.
- Default Case storage is the sibling `../cases/` folder, independent of the launch working directory.
- Store other local market-data files under `../market-data/`; that folder is also excluded from Git.
- Core research behavior is unchanged from v2.8; this release establishes the repository layout for future iterations.


## v2.8 — Cursor-following OHLC + candle change %

- Analyzer and Pattern Viewer OHLC information now follows the candle currently under the crosshair instead of always showing the latest rendered candle.
- Crosshair X continues to snap to the nearest revealed candle, so OHLC never reads unrevealed future data during Replay.
- Added candle `漲跌` to the right of OHLC, calculated as `(Close - Open) / Open × 100`, displayed with an explicit `+` / `-` sign.
- The displayed candle follows the active view timeframe (M1 / M5 / M15 / H1); a currently-forming aggregated candle keeps the existing `[forming]` marker.
- When the cursor leaves the chart, the OHLC row returns to the latest rendered/revealed candle.
- No Case JSON, Drawing, Replay persistence, Pattern, Note, Order, Enricher, or Viewer drawing-interaction schema changes.


## v2.7 — Structural Viewer Drawing non-interactive mode

- Reworked Viewer Drawing read-only behavior from the v2.6 post-render lock approach to a real `drawing_interaction_enabled=False` construction mode inside the shared `ChartWidget`.
- Pattern Analyzer still uses `drawing_interaction_enabled=True`; its Drawing creation, selection, move, resize, templates, clipboard, Magnet, Shift constraints, context menus, and Undo/Redo behavior are unchanged.
- Pattern Viewer now creates saved Drawings non-interactively from the outset:
  - Horizontal Line is non-movable and does not accept mouse buttons.
  - Trend Line is non-movable; native PyQtGraph handles are disabled/hidden and no custom adjustment markers are installed.
  - Rectangle and Text Box keep their existing visual renderer but do not install resize handles or edit callbacks.
  - Fibonacci displays its level lines only; Viewer does not create draggable anchor ROIs, anchor markers, or selection hit strips.
- Viewer does not register Drawing hit-test targets, cannot select Drawings, cannot open Drawing context menus, and all Drawing mutation/sync/clipboard/delete paths have non-interactive guards.
- Removed the v2.6 Viewer `_lock_all_drawing_items()` / `setEnabled(False)` post-render locking layer to avoid old ROI interaction state being reactivated later.
- Permanent Viewer Measure remains unchanged: right-button drag measures directly without Drawing items competing for mouse events.
- Added source-level regression checks for the Viewer non-interactive construction path.

## v2.6 — Viewer Drawing interaction hard-disable + Analyzer Current Range

- Viewer keeps the existing Drawing renderer but removes saved Drawings from the interaction layer after every render/rebuild: no selection, drag, resize, context-menu editing, copy/paste, or delete.
- Analyzer Drawing editing remains unchanged.
- Analyzer toolbar adds `目前 Range`, calculated from revealed raw M1 rows from `replay_start` through the current replay timestamp.
- Current Range points = highest High - lowest Low; Current Range % = Range points / first M1 Open at or after `replay_start` × 100.
- It is independent of M1/M5/M15/H1 view timeframe, updates on replay forward/backward/reset/render, and is transient only.

## v2.5 — Generator create-only safety

- Case Generator is now structurally **create-only** for existing research dates. Existing Case JSON files are skipped and are never opened for writing.
- Existing Analyzer/Viewer research data such as `drawings[]`, `patterns[]`, `intraday_notes[]`, orders, replay state, and enriched reference data therefore remain untouched when Generator is rerun across an old date range.
- Weekend Cases previously renamed by Enricher to `YYYY-MM-DD(W).json` are recognized as the same research date, preventing Generator from creating a duplicate `YYYY-MM-DD.json`.
- New Case files use OS-level exclusive-create mode (`x`), so a destination that appears between the initial scan and the write is still refused rather than overwritten.
- Generator output now reports separate `[CREATE]` and `[SKIP]` rows with created/skipped counts.
- Generator UI shows an explicit safety notice: existing Cases are skipped and research records are not overwritten.

## v2.4 — Broker-aware RTH completeness for CFD maintenance gaps

- Fixed Enricher falsely rejecting otherwise complete NAS100 CFD RTH sessions during New York standard time when the broker maintenance window removes the final ~10 minutes before the theoretical 16:00 cash-session close.
- The theoretical RTH definition remains `09:30–16:00 America/New_York`; it is **not** shortened to 15:49.
- RTH validity now requires **>=95% bar coverage** and allows at most **15 minutes of unquoted tail** after the final available bar.
- A typical winter NAS100 CFD session with 380/390 M1 bars and a 15:49 NY final bar is accepted; genuine early-close or heavily truncated sessions remain rejected.
- Previous RTH and 20D volatility use the same completeness policy.
- Enriched RTH payloads now record `expected_bar_count`, `coverage_ratio`, `tail_gap_minutes`, `min_coverage`, and `max_tail_gap_minutes` for auditability. 20D session rows retain the same completeness diagnostics.
- RTH calculator version is bumped to `1.2`; 20D volatility calculator version is bumped to `1.1`, so existing older payloads are automatically treated as outdated and recalculated even when **only missing/outdated** remains checked.
- Enricher logs now show accepted session bar coverage and tail gap, and no-data messages state the active 95% / 15-minute validity policy.
- Regression tested against the supplied `NAS100_M1_2025-01-01_2026-09-30.txt`: winter sessions such as 2025-11-03 and 2026-02-20 are accepted at 380/390 bars with a 10-minute tail; summer/DST sessions remain 390/390 with zero tail; 20-session volatility succeeds in both regimes.









## v2.3 — Viewer Pattern editor + permanent Measure

- Pattern Viewer left sidebar adds **Case Pattern 編輯** for the currently selected Case.
- Viewer can add, rename, and delete `patterns[]` while Drawings, Notes, Orders, replay research state, and chart geometry remain non-editable.
- Pattern edits use a dedicated atomic pattern-only JSON write path; Viewer still does not call `CaseRepository.save()` for chart state.
- Pattern Filter counts and Case List text are rescanned immediately after a Pattern edit. If the current Case stops matching an active filter, Viewer moves to the next visible Case.
- Viewer Measure is permanently enabled. There is no middle-click toggle in Viewer.
- Any right-button press/drag inside the chart immediately measures endpoint price and percentage change; releasing clears the temporary line/label and the next right-drag is ready immediately.
- The toolbar keeps the `MEASURE` badge visible as an affordance for permanent measure mode.

## v2.2 — Pattern Viewer Case List readability fix

- Fixed Pattern Viewer Case List rows becoming light/white under some Windows + Qt themes while retaining light text.
- Case List now explicitly defines dark normal and alternate-row backgrounds.
- Unselected case text is forced to high-contrast near-white.
- Hover and selected rows now use explicit dark-blue backgrounds with white text.
- Viewer filtering, read-only behavior, chart rendering, Pattern logic, and Case JSON remain unchanged.


## v2.1 — Read-only Pattern Viewer + Pattern-filtered Case Library

- Added a fourth independent program: `run_pattern_viewer.py`.
- Pattern Viewer recursively scans a selected Case folder and reads Case JSON metadata without modifying files.
- The left Pattern Library automatically collects every unique `patterns[].text` tag found in valid Case JSON files.
- Pattern filters support multi-select with:
  - `ANY`: show a Case when it contains at least one selected Pattern.
  - `ALL`: show a Case only when it contains every selected Pattern.
- Selecting a Case immediately loads its market data and reveals the complete research window from `data_start` through `default_end`; there is no Replay stepping in Viewer.
- Viewer reuses the existing chart rendering for candles, saved Drawings, Previous RTH H/L/C, intraday-note callouts, filled order markers, and N-day volatility summary.
- Viewer is structurally read-only:
  - no Drawing creation tools;
  - no Drawing selection/context menus;
  - no move/resize;
  - no Drawing clipboard;
  - no Note/Pattern/Order editors;
  - no Case autosave or `CaseRepository.save()` path.
- View-only operations remain available: pan/zoom, timeframe, X-axis interval, timezone, Auto/AutoAll, screenshot, Drawing visibility, Note-callout visibility, RTH visibility, and volatility N.
- Added `READ ONLY` status badge so Viewer is visually distinct from Pattern Analyzer.


## v1.20 — Alternating Note Callouts + Analyzer N-day volatility summary

- **顯示全部紀錄** now sorts visible notes by time and places them deterministically Top / Bottom / Top / Bottom.
- Each bulk Note callout keeps a slanted leader line and adds a visible yellow/white anchor dot at the exact M1-close anchor.
- Bulk labels stay in the upper/lower outer safe zones instead of occupying the candle area.
- Pattern Analyzer toolbar adds **波動 N** with N restricted to 2–20.
- N-day statistics use only the 20 enriched session rows already stored in `reference_statistics.intraday_volatility_20d.sessions`; Analyzer does not rescan raw M1 data or redefine RTH sessions.
- Displays: `Median Range %`, `1× Std Range %`, `2× Std Range %`, and `3× Std Range %`. Std remains sample standard deviation (`ddof=1`).

## v1.19 — Enricher 20D RTH intraday volatility statistics

- Added `reference_statistics.intraday_volatility_20d` to Case JSON.
- Case Enricher calculates the 20 most recent complete RTH sessions strictly before the Case date.
- Daily range points = `RTH High - RTH Low`.
- Daily range percent = `(RTH High - RTH Low) / RTH Open * 100`.
- Stores and displays 20D median range and sample standard deviation (`ddof=1`) in both points and percent.
- Stores the 20 underlying session rows so later research can derive custom reasonable-volatility bands without rereading raw data.
- Weekend Cases still do not receive Previous RTH reference levels, but 20D volatility statistics are calculated from the preceding 20 valid RTH sessions.
- No volatility lines or overlays are added to Pattern Analyzer in this version.


## v1.18 — Bulk intraday-note callout auto layout

- Reworked **顯示全部紀錄** into a two-zone layout: callouts can use both the top and bottom safe areas instead of stacking only at the top.
- A note anchored in the upper half of the price viewport prefers the bottom zone; a note anchored in the lower half prefers the top zone, reducing leader-line crossings through candles.
- Each side has multiple lanes. Nearby labels are packed into the nearest non-overlapping lane based on estimated on-screen label width.
- When a lane would collide, the layout first tries another lane, then a small horizontal shift, then the opposite side before allowing a dense fallback.
- Bulk mode uses slightly more compact wrapping while single-note Callout behavior is unchanged.
- Layout is transient UI state only; Note JSON, M1 anchors, Undo/Redo, and Drawing data are unchanged.

## v1.17 — Global Drawing / Intraday Note visibility

- Added a chart-toolbar **顯示圖形** checkbox. Unchecking it hides every Drawing view object without deleting or mutating `drawings[]`; re-checking restores them.
- Added a **顯示全部紀錄** checkbox in the intraday-note panel. When enabled, all revealed notes whose M1 anchors are inside the current X viewport are drawn together as callouts; when disabled, bulk callouts are removed and the note selection is cleared.
- Existing single-note click Callout behavior remains available after bulk mode is turned off.
- Both visibility controls are transient UI state and are not written to Case JSON / Undo history.

## v1.16 — Adjustment handle interaction fix

- Restored native PyQtGraph ROI handles to full opacity so they remain reliable mouse hit targets.
- Keeps the shared FIBO-style circular adjustment markers as the visible handle style.
- Fixes Trend Line / Rectangle / Text Box handles behaving like whole-object move instead of resize.
- No changes to FIBO anchor logic, Undo/Redo, Magnet, templates, or Drawing persistence.

## v1.15 — Unified adjustment-point style

- Trend Line endpoint handles, Rectangle side resize handles, and Text Box resize handles now use the same visible style as Fibonacci anchors.
- Shared style: circular 12 px marker, dark fill, blue 2 px outline.
- Native PyQtGraph ROI handles remain as invisible interactive hit targets, so existing drag/resize behaviour is preserved.
- Adjustment markers are visible only while the drawing is selected.
- Horizontal Line is unchanged because it has no separate resize anchor.

## v1.14 Fibonacci two-anchor editing

- FIBO 未選取時不顯示任何調整點。
- 雙擊選取 FIBO 後，只顯示兩個圓形 Anchor：0 與 1。
- 0 Anchor 直接控制 `start.time + start.price`；1 Anchor 直接控制 `end.time + end.price`。
- 兩個 Anchor 都可自由上下左右拖曳，因此可同時修改 X（時間）與 Y（價格）。
- Ctrl Magnet 保留，可在拖曳 Anchor 時吸附目前已揭露 K 棒的 OHLC。
- FIBO levels 仍由 0/1 兩個基準自動重新計算。
- v1.12 新增的整體線型與線粗設定保留。
- 選取 FIBO 不再把整組線改成黃色；兩個 Anchor 本身就是選取提示。
- 每條 level 額外保留不可見的寬 hit-zone，方便雙擊選取與右鍵設定，但不會顯示額外控制點。
- 一次 Anchor 拖曳仍只建立一筆 Undo。

## v1.12 Fibonacci line controls + line style

- Fibonacci 不再顯示任何菱形／調整點；所有調整改由水平 level 線本身完成。
- Level `0` 線：調整第一個基準 Y（start price）。
- Level `1` 線：調整第二個基準 Y（end price）。
- 畫面上「最高價格」的 Fibonacci level 線：調整整組 Fibonacci 的右側 X 範圍。
  - 若最高價格線同時是 0 或 1，拖曳以螢幕方向判斷：垂直拖曳調 Y、水平拖曳調 X。
  - 因此由高往低畫、levels 為 `0, 0.5, 1, 2, 3` 時，最高價格通常是 `0`，X 只由 `0` 線調整。
- Fibo 設定新增「整體線條」：
  - 線型：實線 / 虛線 / 點線
  - 線粗：1–12
- 線型與線粗會套用到所有 Fibonacci levels，並隨 Fibonacci Template 儲存。
- 保留每一個 level 各自的顏色設定。
- Ctrl magnet 仍支援 0/1 的 Y 調整與最高價格線的 X 調整。
- Fibo 操作完成仍只產生一筆 Undo transaction。

## v1.11 Note Callout timeframe X alignment

- Note Callout 的 Y anchor 仍固定使用該 replay_time 的原始 M1 close。
- X anchor 改為依目前顯示 timeframe 對齊「包含該分鐘的 K 棒起始時間」。
- 例：12:59 Note 在 M1 顯示於 12:59；在 M5 顯示於 12:55；在 M15 顯示於 12:45。
- 對齊規則與 aggregate_visible_bars() 使用相同的 bucket convention，避免 Callout 指到下一根 K。


## v1.10 Selected Intraday Note Callout
- 單擊一筆盤中紀錄後，以該紀錄時間對應的原始 M1 Close 作為 Anchor。
- 從 Anchor 拉出高 z-order 黃色斜線，連到圖表上方固定安全區。
- 上方顯示深色背景／白字的盤中紀錄文字；一次只顯示目前選取的一筆。
- 切換 Note、取消選取、切換 Case 時同步更新／清除。
- Callout 為 transient overlay，不寫入 drawings[]、不進 Template、也不加入 Undo/Redo。
- 不新增 Callout 專用 JSON 欄位；既有 replay_time 直接作為 M1 Anchor 的時間來源。

# Pattern Research v1.5

## v1.5 Previous RTH workflow
- Adds a separate `Case Enricher` program (`run_case_enricher.py`).
- Weekday cases: finds the previous complete 09:30-16:00 America/New_York RTH session and writes H/L + high/low timestamps into `reference_levels.previous_rth`.
- Weekend cases: do not receive Previous RTH data, are marked `calendar.is_weekend=true`, and the JSON filename receives `(W)`.
- Pattern Analyzer only reads the stored JSON reference levels. Changing `display.timezone` never changes the RTH values.
- The right Research panel has a `Previous RTH H/L` checkbox. Cases without enriched data show `無 RTH 資料` and the checkbox is disabled.

# Pattern Research v1.4

## v1.4 time-model changes
- Adds `time_context.case_timezone` and `time_context.canonical_timezone`.
- Display timezone is independent from the case timezone.
- Case Generator now guarantees chronological A -> B -> C timestamps across midnight.
- Example: A=08:00, B=09:30, C=04:30 produces C on the next calendar day.

# Pattern Research v1.3

## v1.3 changes
- Rectangle resize controls now use four edge-midpoint handles: left, right, top, and bottom.
- Corner resize handles remain disabled.
- Text box resize behavior is unchanged (right-middle and bottom-middle only).

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


## v1.6 Enricher upgrade

Previous RTH now stores High / Low / Close plus their timestamps. The Enricher only skips an existing payload when all required fields are present and the calculator version, session timezone/start/end, and data source ID match the current settings. Older H/L-only payloads are automatically recalculated and upgraded.

## v1.7 - Temporary Measure Mode

- Middle-click toggles Measure Mode on/off.
- While Measure Mode is enabled, press and hold the right mouse button to set the start point and drag the endpoint.
- The temporary line and endpoint label update continuously while dragging.
- Label shows endpoint price and percentage change from the start price.
- Releasing the right mouse button immediately removes the measurement; Measure Mode stays enabled for the next measurement.
- Middle-click again exits Measure Mode.
- Measurements are transient only and are never written to Case JSON / drawings[].

## v1.8

- Intraday Note supports inline edit: double-click a note to edit in place; Save/Cancel buttons appear only on that row.
- Added global research Undo/Redo: Ctrl+Z / Ctrl+Y.
- Undoable research state includes drawings, patterns, and intraday notes only. Replay/view state and RTH/Measure/UI state are excluded.
- Drawing operations covered: add, delete, paste, style edit, move/resize, Fibonacci edits, Text Box move/resize.
- Pattern operations covered: add/delete.
- Intraday Note operations covered: add/delete/edit.
- Text editors retain native text undo/redo while focused.
- Trend Line whole-object movement now persists absolute endpoints correctly so moved/copied lines do not jump back after re-render.


### v1.10 — Intraday Note Callout deselection
- Click the currently selected intraday note again to clear its selection and hide the chart callout.
- Press Esc in the analyzer window to clear the selected intraday note and hide the callout.
- Double-click inline note editing remains available.


## v1.14 — Geometry Picker
- Drawing 雙擊選取改用統一的 screen-pixel Geometry Picker（8 px tolerance）。
- Trend/Horizontal Line：只依實際線幾何判定。
- Rectangle：只依四條邊判定，不以內部填滿區域搶選取。
- Fibonacci：只依各條可見 level 線段判定，不再使用整個包覆區域。
- Text Box：框內仍可直接選取。
- 多物件靠近時以游標到實際圖形的 pixel distance 最近者優先；完全同距離才以較後建立者作 tie-break。
- 右鍵選單、Fibo Anchor、Drawing 移動/Resize、Undo/Redo、Template 行為不變。
