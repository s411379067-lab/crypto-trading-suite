#### Pattern Replay Research Platform — Architecture Specification v0.1

##### 1. 專案目標

本系統的核心用途不是交易下單，也不是策略自動回測，而是針對歷史市場資料進行「逐步時間回放、圖表標註、盤中紀錄與 Pattern 研究」。

整體工作流程分成兩個獨立程式：

**Program A — Case Generator**

負責從歷史行情資料中，依照使用者指定的商品、日期與時間條件，批次建立研究 Case JSON。

**Program B — Pattern Analyzer**

負責讀取 Case JSON 與原始歷史行情資料，渲染交易圖，進行多週期 K 線回放、Drawing、盤中紀錄與 Pattern 標記，並將研究結果持續寫回原 Case JSON。

兩個程式彼此獨立，但必須共用相同的 `shared-core`，避免 JSON 格式、時間處理、行情解析與資料驗證邏輯各自發展。

---

#### 2. 核心架構原則

##### 2.1 原始行情與研究 Case 必須分離

原始行情資料是市場資料來源，應視為唯讀。

Case JSON 不複製整段 OHLC 歷史資料，而是記錄：

- Case 使用哪個商品
- 需要哪個歷史資料來源
- Case 的時間邊界
- Replay 設定
- Pattern
- 盤中紀錄
- Drawings
- 使用者研究狀態

因此：

```text
Market Data = 原始行情事實
Case JSON   = 研究設定 + 研究結果
```

這能避免同一份行情被複製到大量 JSON 中，也避免未來行情資料修正後產生多份不同版本。

---

##### 2.2 JSON 是 Case 的主要 Source of Truth

每個 Case 都是一個可獨立搬移、備份、同步的 JSON。

Case Library 可以透過掃描 Windows 資料夾建立。

未來即使加入 SQLite、DuckDB 或其他 Index Database，也只能作為：

```text
Search Index
Cache
Statistics Index
```

而不能取代 Case JSON 本身。

---

##### 2.3 Domain 不依賴 UI、Chart Library 或 Database

核心研究邏輯不可直接依賴：

```text
React
KLineChart
Lightweight Charts
SQLite
Windows UI
```

依賴方向應維持：

```text
UI
↓
Application
↓
Domain
↑
Adapters
```

例如 Pattern、Replay、Drawing 的核心資料模型不能含有 React component、Canvas pixel 或特定 Chart Library object。

這是防止後續功能擴充時整個架構倒置的主要規則。

---

#### 3. 專案結構

建議 repository 結構：

```text
pattern-research/
│
├─ apps/
│  │
│  ├─ case-generator/
│  │  ├─ ui/
│  │  ├─ application/
│  │  └─ adapters/
│  │
│  └─ pattern-analyzer/
│     ├─ ui/
│     ├─ application/
│     └─ adapters/
│
├─ packages/
│  │
│  ├─ shared-core/
│  │  ├─ case/
│  │  ├─ market-data/
│  │  ├─ replay/
│  │  ├─ time/
│  │  ├─ timeframe/
│  │  ├─ pattern/
│  │  ├─ drawing/
│  │  ├─ notes/
│  │  └─ validation/
│  │
│  ├─ market-data-adapters/
│  │  ├─ txt/
│  │  ├─ csv/
│  │  └─ parquet/
│  │
│  └─ chart-adapters/
│     └─ ...
│
└─ docs/
   └─ architecture.md
```

Program A 與 Program B 不應直接引用彼此。

兩者只依賴共同的 `shared-core` 與必要 Adapter。

---

#### 4. Case Definition

一個 Case 定義為：

> 某商品、某研究日期、某段指定時間範圍的一次 Pattern 研究樣本。

例如：

```text
NAS100
2026-09-16
A = 20:00
B = 21:30
C = 23:00
```

Case 不等於完整交易日，也不等於一筆交易。

Case 的主要目的是描述「這一次要研究的歷史市場片段」。

---

#### 5. Case 時間模型

每個 Case 必須至少包含三個核心時間點：

```text
A = data_start
B = replay_start
C = default_end
```

它們的意義如下：

```text
A ───────────── B ───────────── C
│               │               │
│ 初始可見資料   │ Replay 起點    │ 預設研究終點
│               │               │
└──── 初始視圖 ──┘               │
                └──── Replay ────┘
```

第一次開啟 Case 時：

```text
Visible Range = A → B
```

未來資料 B 之後不可見。

每次 Replay 前進後：

```text
Visible Range = A → Current Replay Time
```

C 只是「預設研究終點」，不是硬限制。

Replay 到 C 後仍可繼續往後，只要 Market Data Source 中存在後續資料。

因此 Case 必須區分：

```text
default_end
current_replay_time
```

不可因使用者繼續向後播放而覆寫原本的 C。

---

#### 6. Market Data Contract

##### 6.1 標準 Bar

所有 Market Data Adapter 最終都必須輸出統一格式：

```ts
interface Bar {
    timestamp: number
    open: number
    high: number
    low: number
    close: number
    volume?: number
    tickVolume?: number
    spread?: number
}
```

`timestamp` 必須是 canonical timestamp。

核心模組不能知道原始資料來自 TXT、CSV、Parquet 或 API。

---

##### 6.2 MarketDataProvider

統一介面概念：

```ts
interface MarketDataProvider {
    getAvailableRange(symbol, resolution): TimeRange
    getBars(symbol, resolution, start, end): Bar[]
}
```

Pattern Analyzer 不直接解析 TXT。

必須透過 Adapter：

```text
TXT
↓
TxtMarketDataAdapter
↓
MarketDataProvider
↓
Replay / Chart / Aggregator
```

未來如果改為 Parquet：

```text
Parquet
↓
ParquetMarketDataAdapter
↓
MarketDataProvider
```

Replay 與 Pattern Analyzer 不需要修改。

---

#### 7. 多週期與 Replay 模型

系統必須把兩個概念完全分開：

```text
View Timeframe
Replay Step
```

例如：

```text
View Timeframe = M5
Replay Step    = 1 minute
```

這代表使用者看的是 5 分 K，但每按一次 Replay Forward，只增加 1 分鐘資訊。

---

##### 7.1 最細資料解析度

Replay 的最小步進不能小於底層資料實際解析度。

例如：

```text
資料最細 = M5
```

則不允許真實 Replay：

```text
Step = M1
```

因為 M5 OHLC 無法還原 M5 內部 5 個 1 分鐘的真實走法。

如果需要：

```text
View = M5
Step = M1
```

則 Market Data 至少需要 M1。

---

##### 7.2 正在形成中的 K 棒

當：

```text
View = M5
Replay Step = M1
```

Replay 時間從：

```text
21:30
→ 21:31
→ 21:32
→ 21:33
...
```

圖上的 21:30～21:35 M5 K 棒必須即時變化。

例如 21:32 時：

```text
Open  = 21:30 M1 Open
High  = max(21:30 ~ 21:32)
Low   = min(21:30 ~ 21:32)
Close = 21:32 M1 Close
```

不能提前使用 21:33～21:34 的資訊。

---

##### 7.3 Timeframe Aggregator

多週期視圖不應要求每個 timeframe 都有獨立原始檔。

建議：

```text
最細原始資料
↓
Timeframe Aggregator
↓
M1 / M5 / M15 / M30 / H1 ...
```

Aggregator 只允許使用：

```text
timestamp <= replay_current_time
```

的資料。

此規則必須在資料層或 Replay Domain 層強制執行，而不能只靠 UI 隱藏。

---

#### 8. Time Model

原始資料可能同時存在：

```text
UTC+3
UTC+8
```

但系統核心不能以 UTC+3 或 UTC+8 作為永久時間基準。

所有資料進入 Market Data Layer 後統一轉成：

```text
Canonical UTC Timestamp
```

原始時區只保存為 metadata。

圖表 View 可以自由選擇顯示時區，例如：

```text
UTC
UTC+3
UTC+8
America/New_York
Asia/Taipei
```

Drawing、Pattern、Note 與 Replay 都以 canonical timestamp 儲存。

因此切換顯示時區不會改變 Case 資料。

---

#### 9. Replay Contract

Replay Engine 只負責「現在市場走到哪裡」。

核心狀態：

```ts
interface ReplayState {
    replayStart: number
    currentTime: number
    stepMinutes: number
    status: "idle" | "running" | "paused"
}
```

Replay Engine 不負責：

```text
畫 K 線
畫 Drawing
存 Pattern
存 Note
解析 TXT
```

它只改變：

```text
currentTime
```

其他模組依據 currentTime 決定自己可以看到多少資訊。

---

##### 9.1 Visibility Boundary

所有研究模組必須遵守：

```text
visible_timestamp <= replay.currentTime
```

這條限制不只套用在 Chart。

未來如果增加：

```text
Indicator
Auto Pattern Detection
Statistics
AI Analysis
Reference Levels
```

同樣不能讀取 Replay 時點之後的資料。

避免 look-ahead bias。

---

#### 10. Pattern Model

Pattern 不是單一 Enum，也不是固定的 A/B/C/D。

一個 Case 可以同時存在多種 Pattern，而且 Pattern 採自由文字描述，例如：

```text
開盤TR
開盤TR BO 1R後反轉
```

因此資料結構採集合：

```ts
interface PatternTag {
    id: string
    text: string
    createdAt: number
    updatedAt: number
}
```

Case：

```ts
patterns: PatternTag[]
```

Pattern 與一般 Note 必須分開。

Pattern 是可以：

```text
搜尋
篩選
統計
建立 Dataset
```

的正式研究標籤。

---

##### 10.1 Pattern 未來擴充

目前先採自由文字。

但架構上 PatternTag 要有穩定 ID，而不能只存純 string array。

原因是未來可能新增：

```text
Pattern Alias
Pattern Group
Pattern Version
Pattern Relationship
Pattern Rename
Statistics Index
```

這些功能都需要穩定 ID。

---

#### 11. Intraday Notes

盤中紀錄與 Pattern 分離。

盤中紀錄必須保存「當時 Replay 走到哪裡」。

例如：

```ts
interface IntradayNote {
    id: string
    replayTime: number
    text: string
    createdAt: number
    updatedAt: number
}
```

這代表重新打開 Case 時可以知道：

> 使用者在只看到 21:42 市場資訊的情況下，當時記錄了什麼。

不能只把所有內容存在一個最終大文字框裡。

---

#### 12. Drawing Contract

Drawing 是 Case 的永久研究資料。

所有 Drawing 必須：

```text
可以保存
可以重新載入
可以拖曳
可以修改
可以刪除
```

Drawing 不能儲存 Canvas pixel：

```text
x = 523
y = 341
```

必須使用市場座標：

```text
timestamp
price
```

例如：

```ts
interface DrawingPoint {
    time: number
    price: number
}
```

Trend Line：

```ts
{
    id,
    type: "trend_line",
    points: [
        {time, price},
        {time, price}
    ],
    style: {...}
}
```

Rectangle：

```ts
{
    id,
    type: "rectangle",
    points: [
        {time, price},
        {time, price}
    ],
    style: {...}
}
```

這樣 Drawing 不依賴螢幕尺寸，也不依賴特定 Chart Library。

---

##### 12.1 Drawing 與 Timeframe 解耦

Drawing 不屬於某個固定 timeframe。

例如使用者在 M5 畫：

```text
time = 21:40
price = 25500
```

切換到 M1 或 M15 時仍然應該看到同一個 Drawing。

除非未來另外加入：

```text
visible_on_timeframes
```

功能，否則 Drawing 預設跨 timeframe 共用。

---

#### 13. Template Contract

Template 模組目前只預留介面，不在 v0.1 鎖定具體欄位。

未來可以逐步控制：

```text
Chart Appearance
Indicators
Default View Timeframe
Replay Step
Workspace Layout
Toolbar
Reference Levels
```

但 Template 不得改變 Case Domain 的核心定義。

Template 應該是可替換設定，而不是研究資料本身。

---

#### 14. Case Library

Pattern Analyzer 以 Windows 資料夾作為 Library 根目錄。

例如：

```text
Cases/
├─ NAS100/
│  ├─ 2026/
│  │  ├─ 09/
│  │  │  ├─ 2026-09-14.json
│  │  │  ├─ 2026-09-15.json
│  │  │  └─ ...
│
├─ SP500/
└─ US30/
```

程式掃描資料夾中的 Case JSON，再顯示成 Case Library。

使用者仍然可以在 Windows Explorer 中：

```text
新增資料夾
移動 Case
重新分類
備份
同步
```

Pattern Analyzer 必須能重新掃描並反映改變。

---

##### 14.1 Future Index

當 Case 數量增加到數千或數萬後，可以增加：

```text
CaseIndex
```

用於快速搜尋：

```text
symbol
date
patterns
timeframe
createdAt
```

但 Index 必須可由 JSON 全部重新建立。

因此：

```text
JSON = Truth
Index = Cache
```

---

#### 15. Case JSON Schema v0.1

概念結構如下：

```json
{
  "schema_version": "0.1",

  "case": {
    "id": "uuid",
    "symbol": "NAS100",
    "research_date": "2026-09-16"
  },

  "market_data": {
    "source_id": "nas100-primary",
    "source_type": "txt",
    "resolution": "M1"
  },

  "time_range": {
    "data_start": "A",
    "replay_start": "B",
    "default_end": "C"
  },

  "display": {
    "view_timeframe": "M5",
    "timezone": "Asia/Taipei"
  },

  "replay": {
    "step_minutes": 1,
    "current_time": "B"
  },

  "patterns": [],

  "intraday_notes": [],

  "drawings": [],

  "metadata": {
    "created_at": "",
    "updated_at": ""
  }
}
```

此 JSON 只是 v0.1 Contract。

實作前仍需正式建立 JSON Schema validation。

---

#### 16. Case Generator Responsibilities

Case Generator 只負責：

```text
選擇 Market Data Source
選擇 Symbol
選擇日期範圍
設定 A / B / C
批次建立 Case JSON
驗證原始資料是否存在
```

Case Generator 不應包含：

```text
Replay
Chart
Drawing
Pattern Analysis
Statistics
```

它的輸出就是符合 Case Schema 的 JSON。

---

#### 17. Pattern Analyzer Responsibilities

Pattern Analyzer 負責：

```text
Case Library
讀取 Case JSON
解析 Market Data
多週期圖表
Replay
Partial Candle
Drawing
Intraday Notes
Pattern Tags
儲存 Case
```

Pattern Analyzer 不負責批次定義大量研究日期。

---

#### 18. Shared Core Responsibilities

`shared-core` 是整套系統最重要的穩定層。

至少包含：

```text
Case Schema
Case Validation
Time Model
Market Data Contract
Timeframe Aggregator
Replay State
Pattern Model
Intraday Note Model
Drawing Model
Schema Migration
```

兩個 App 都只能透過 shared-core 理解 Case。

---

#### 19. Adapter Boundary

至少保留三種 Adapter Boundary。

##### Market Data Adapter

```text
TXT / CSV / Parquet / API
↓
MarketDataProvider
```

##### Chart Adapter

```text
Domain Bar / Drawing
↓
ChartAdapter
↓
實際 Chart Library
```

##### File / Repository Adapter

```text
Case Domain
↓
CaseRepository
↓
Windows File System
```

如此未來替換任何一層都不應修改 Domain。

---

#### 20. CaseRepository

Pattern Analyzer 不應散落大量：

```text
open()
read()
write()
json.load()
json.dump()
```

檔案操作。

統一使用：

```ts
interface CaseRepository {
    load(path): ResearchCase
    save(case): void
    list(root): CaseSummary[]
    validate(path): ValidationResult
}
```

未來增加 Auto Save、Backup、Cloud Sync 時只需要修改 Repository / Adapter。

---

#### 21. Auto Save 與資料安全

Pattern Analyzer 未來應採：

```text
修改 Domain
↓
Debounced Auto Save
↓
Temporary File
↓
Atomic Replace
```

不要直接覆寫原 JSON。

建議：

```text
case.json
case.json.tmp
```

寫入成功後才 atomic replace。

避免程式崩潰導致整個 Case JSON 損壞。

---

#### 22. Schema Versioning

所有 Case JSON 必須有：

```json
"schema_version": "0.1"
```

未來 JSON 欄位改變時：

```text
v0.1
↓
Migration
↓
v0.2
```

Pattern Analyzer 不能直接假設所有舊 JSON 都符合最新版格式。

需要：

```text
CaseMigrator
```

這是避免後續迭代破壞舊研究資料的重要機制。

---

#### 23. Event Model

Pattern Analyzer 內部建議使用 Domain/Application Event 解耦功能。

例如：

```text
CASE_OPENED
REPLAY_STARTED
REPLAY_ADVANCED
VIEW_TIMEFRAME_CHANGED
DRAWING_CREATED
DRAWING_UPDATED
DRAWING_DELETED
NOTE_CREATED
PATTERN_CREATED
CASE_SAVED
```

例如新增 Pattern 時：

```text
UI
↓
AddPatternCommand
↓
Pattern Domain
↓
PATTERN_CREATED
```

未來 Search Index、Statistics、Auto Save 可以訂閱事件，不必直接修改 Pattern 模組。

---

#### 24. 禁止的架構

以下形式應避免：

```text
Chart Component
↓
直接讀 TXT
↓
直接修改 JSON
↓
Replay
↓
Pattern
```

以及：

```text
React State = Case Data
```

也不能：

```text
Drawing = Chart Library Object
```

或：

```text
Replay = Chart current candle index
```

這些都會造成 UI、資料與研究邏輯高度耦合。

---

#### 25. 正確的主要資料流

開啟 Case：

```text
Case Library
↓
CaseRepository
↓
ResearchCase
↓
MarketDataProvider
↓
Replay Engine
↓
Timeframe Aggregator
↓
Chart Adapter
↓
Chart
```

使用者建立研究資料：

```text
User Action
↓
Application Command
↓
Domain
↓
ResearchCase Updated
↓
Auto Save
↓
JSON
```

---

#### 26. 後續功能擴充方向

目前架構應允許未來新增，但 v0.1 不實作：

```text
Indicator Engine
Reference Levels
Statistics
Pattern Search
Pattern Comparison
Screenshot Export
Multi Chart
Multi Symbol
AI Pattern Detection
Automatic Feature Extraction
Order Simulation
Strategy Backtest
```

這些未來應以新模組方式加入，而不是修改 Case / Replay / Market Data 的核心責任。

---

#### 27. v0.1 開發優先順序

第一階段先完成 shared-core：

```text
Case Schema
Time Model
MarketDataProvider
Txt Adapter
Replay Engine
Timeframe Aggregator
Drawing Model
Pattern Model
Note Model
CaseRepository
```

第二階段完成 Pattern Analyzer 最小流程：

```text
讀 Case
↓
讀歷史資料
↓
顯示 A~B
↓
每次前進 1 分鐘
↓
Partial M5 Candle 即時更新
↓
切換 timeframe
↓
Drawing
↓
Intraday Notes
↓
Pattern
↓
Save
```

第三階段再完成 Case Generator。

---

#### 28. v0.1 成功標準

第一版只要能穩定完成以下完整流程，就代表核心架構成立：

```text
建立 NAS100 Case JSON
↓
Pattern Analyzer 開啟 JSON
↓
讀取 M1 歷史行情
↓
初始顯示 A~B
↓
View = M5
Replay Step = M1
↓
逐分鐘前進
↓
正在形成的 M5 K 棒即時更新
↓
切換 M1 / M5 / M15
↓
畫圖
↓
新增盤中文字紀錄
↓
新增多個自由文字 Pattern
↓
關閉程式
↓
重新開啟
↓
完整還原 Replay、Pattern、Note、Drawing
```

在這條流程完成以前，不應優先加入 Order、PnL、自動交易或複雜統計功能。

---

#### 29. 目前已定案與暫緩項目

已定案：

```text
兩個獨立程式
Case JSON 與 Market Data 分離
多週期
View Timeframe 與 Replay Step 解耦
Partial Candle 即時更新
A / B / C 時間模型
C 可向後延伸
任意顯示時區
Pattern 支援多筆自由文字
盤中紀錄保存 Replay Time
Drawing 完整保存並可再次修改
資料夾式 Case Library
JSON 為 Case Source of Truth
```

暫緩到實作階段：

```text
Template 具體欄位
Chart Library 最終選型
完整 Drawing Tool 清單
UI Layout
Indicators
Statistics
Screenshot Template
```

這些項目現在不應阻塞核心架構實作。

#### v0.1.2 Drawing / Axis additions

- Fibonacci is stored as a serializable Drawing Domain object (`start`, `end`, `levels`, `style`); PyQtGraph ROI/handles/level lines are view-only objects reconstructed from JSON.
- Drawing color UI uses one shared legacy TradingView palette service rather than independent QColorDialog behavior.
- New line drawings default to white.
- Crosshair labels are UI-only state and are never written into Case research data.
- `display.x_tick_interval` controls X-axis label spacing independently from `display.view_timeframe` and Replay step.
