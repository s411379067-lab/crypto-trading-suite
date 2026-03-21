# Trading Journal Desktop App

A comprehensive **Electron-based desktop application** for tracking, analyzing, and visualizing cryptocurrency trading performance. This tool provides traders with powerful filtering, analytics, and charting capabilities to improve trading decisions through data-driven insights.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Application Structure](#application-structure)
- [Features Deep Dive](#features-deep-dive)
- [Data Format](#data-format)
- [Development](#development)
- [License](#license)

## ✨ Features

### Core Functionality
- **📊 Trade Management**: Add, edit, and delete trades with comprehensive data tracking
- **🔍 Advanced Filtering**: Multi-condition filtering with AND/OR logic across all trade attributes
- **📈 Analytics Dashboard**: 50+ performance metrics including P&L, win rate, Sharpe ratio, drawdown analysis
- **📉 Interactive Charts**: 16+ chart types powered by ECharts for visualizing trading performance
- **💾 Excel Integration**: Direct import/export with `.xlsx` files using native Excel formats
- **🎨 Customizable Views**: Flexible column ordering, sorting, and dashboard layouts

### Analytics Metrics Categories

#### 損益總覽 (P&L Overview)
- Total Profit/Loss (Fully Closed & Realized)
- Profit-only and Loss-only PnL

#### 單筆/期間平均 (Average Per Period)
- Average PnL per Trade, Day, Month, Year

#### 平均贏損 (Win/Loss Averages)
- Average Winning/Losing PnL per Trade, Day, Month, Year

#### 極值損益 (Extremes)
- Best/Worst PnL per Trade, Day, Month, Year

#### Drawdown Analysis
- Current Drawdown, Max Drawdown (MDD), Average Drawdown, Ulcer Index

#### 風控/分布 (Risk & Distribution)
- Win Rate, Profit Factor, Payoff Ratio
- Sharpe, Sortino, Calmar ratios
- Standard Deviation, Skewness, Kurtosis
- Tail Ratio, Max Consecutive Wins/Losses

### Chart Types

#### 績效與穩定度 (Performance & Stability)
- R-based Equity Curve (累積)
- Underwater Chart (Drawdown)
- Trade Series (單筆時間序列)
- Rolling Average/Win Rate/Profit Factor

#### 分佈 (Distribution)
- Histogram
- Cumulative Distribution (ECDF)

#### 時間維度 (Time Analysis)
- Time-of-Day Average/Count/Total (by entry time)

#### 分類比較 (Categorical Analysis)
- Strategy × Average/Win Rate/Profit Factor
- Structure × Average
- Context Rating × Average

## 🛠 Tech Stack

- **Framework**: [Electron](https://www.electronjs.org/) - Cross-platform desktop application
- **Frontend**: Vanilla JavaScript (ES6+) with HTML5 & CSS3
- **Charts**: [ECharts 5](https://echarts.apache.org/) - Interactive data visualization
- **Excel Processing**: [SheetJS (xlsx)](https://sheetjs.com/) - Excel file I/O
- **Architecture**: Model-View pattern with modular dashboard system

## 📦 Installation

### Prerequisites
- [Node.js](https://nodejs.org/) (v14 or higher)
- npm (comes with Node.js)

### Setup

1. **Clone or download this repository**
   ```bash
   cd trade-journal-desktop
   ```

2. **Install dependencies**
   ```bash
   npm install electron xlsx
   ```

3. **Verify the data folder exists**
   ```bash
   # The default Excel file location
   data/trading_journal.xlsx
   ```

## 🚀 Usage

### Starting the Application

```bash
npm start
```

This will launch the Electron app and automatically load the default Excel file from `data/trading_journal.xlsx`.

### Application Workflow

1. **Trades Table Tab**
   - View all trades in a spreadsheet-like interface
   - Add new trades with the "+ Add Trade" button
   - Edit cells directly by clicking (supports inline editing)
   - Apply filters to narrow down trades
   - Sort by 損益% (P&L percentage)
   - Open/Save Excel files using the toolbar buttons

2. **Analytics Tab**
   - View comprehensive performance metrics
   - Configure calculation basis (損益金額/總倉損益%/R)
   - Include/exclude fees in calculations
   - Group trades by Product, Timeframe, Side, Strategy, etc.
   - View breakdown tables sorted by various metrics
   - Input account values for money-based metrics

3. **Charts Tab**
   - Visualize trading performance with interactive charts
   - Switch between 4-tile and 6-tile layouts
   - Customize each chart type independently
   - Adjust rolling window and bucket sizes
   - All charts respect active filters from Trades tab

4. **Customize Tab**
   - Test metric calculations
   - Preview chart configurations
   - View metric/chart registry

### Excel Integration

#### Auto-load Default File
On startup, the app automatically loads `data/trading_journal.xlsx`

#### Open Different File
Click **"Open Excel"** button → Select `.xlsx` file → File loads into table

#### Save Changes
Click **"Save Table"** button → Current table data saves back to the opened Excel file

## 📁 Application Structure

```
trade-journal-desktop/
├── index.html              # Main UI (3584 lines)
├── main.js                 # Electron main process
├── preload.js              # IPC bridge (contextBridge)
├── package.json            # Dependencies & scripts
├── data/
│   └── trading_journal.xlsx  # Default data file
└── js/
    ├── dashboard_builder.js   # Dynamic dashboard tile creation
    └── dashboard_registry.js  # Metric & chart definitions
```

### Key Files

#### `main.js`
- Electron app initialization
- File system operations (Excel read/write)
- IPC handlers for:
  - `excel:openDefault` - Load default file
  - `excel:openPath` - Load specific path
  - `excel:open` - Open file dialog
  - `excel:save` - Save current data

#### `preload.js`
- Secure IPC bridge between renderer and main process
- Exposes `window.desktopExcel` API to frontend

#### `index.html`
- Complete single-page application
- Embedded styles and scripts
- Four main sections: Trades, Analytics, Charts, Customize
- Comprehensive filter system
- ECharts integration
- Cell editing with contenteditable and custom dropdowns

#### `js/dashboard_registry.js`
- `METRIC_REGISTRY`: 50+ metric definitions with categories
- `CHART_REGISTRY`: 16+ chart type definitions
- Helper functions for grouped selectors

#### `js/dashboard_builder.js`
- Dynamic tile creation for analytics dashboard
- Metric selection and rendering
- Value formatting utilities

## 🎯 Features Deep Dive

### Filter System
- **Column-based filters**: Select any column as filter criteria
- **Multiple operators**: `=`, `≠`, `>`, `<`, `≥`, `≤`, `contains`, `starts`, `ends`, `between`, `is/is not`
- **Multi-select support**: For columns with multiple tags (strategies, structures)
- **Chips UI**: Google Sheets-like multi-select with dropdown popover
- **Real-time filtering**: Instant updates across all views
- **Filter summary**: Visual indicator showing active filter conditions

### Editable Table
- **Inline editing**: Click any cell to edit directly
- **Smart columns**: 
  - Single-select dropdowns (產品, 週期, 多空)
  - Multi-select with chips (策略, 結構, 建倉依據, 出場)
  - Numeric fields with validation
  - Date/time pickers
- **Drag-to-reorder**: Column headers are draggable
- **Paste support**: Excel/Sheets-style multi-cell paste (tab/newline delimited)
- **Keyboard navigation**: Tab through cells, Enter to confirm

### Analytics Engine
- **Configurable basis**: Calculate on 損益金額, 總倉損益%, or R (risk-reward)
- **Fee handling**: Toggle between gross and net P&L
- **Grouping**: Analyze by product, timeframe, side, strategy, structure, etc.
- **Breakdown tables**: Detailed statistics per group (N, Win%, EV, Sum, PF, Payoff, MDD, Std dev, Sharpe, Sortino, Calmar)
- **Account inputs**: Starting capital, unrealized P&L, current account value
- **Multi-tag support**: Strategies/structures can be comma-separated, contributing to multiple groups

### Chart System
- **Responsive layouts**: Switch between 4-tile and 6-tile grid
- **Per-chart configuration**: Each tile independently selects chart type
- **Full-screen mode**: Expand any chart for detailed analysis
- **Unified settings**: Base metric, fee handling, rolling window, bucket size
- **ECharts integration**: Interactive tooltips, zoom, pan, data export
- **Modular renderers**: Custom chart functions in `CUSTOM_CHART_RENDERERS`

## 📊 Data Format

### Required Columns
The Excel file should contain these columns (Chinese labels):

| Column | Type | Description |
|--------|------|-------------|
| 產品 | Single-select | Trading product (e.g., BTC, ETH) |
| 週期 | Single-select | Timeframe (e.g., 5M, 1H, 1D) |
| 多空 | Single-select | Side (多/空 - Long/Short) |
| 策略 | Multi-select | Trading strategies (comma-separated) |
| 結構 | Multi-select | Market structure tags |
| 建倉依據 | Multi-select | Entry basis tags |
| 出場 | Multi-select | Exit reason tags |
| entry_date | Date | Entry date (YYYY-MM-DD) |
| entry_time | Time | Entry time (HH:MM) |
| exit_date | Date | Exit date (YYYY-MM-DD) |
| exit_time | Time | Exit time (HH:MM) |
| hold_mins | Number | Holding period in minutes |
| 損益金額 | Number | P&L in currency |
| 手續費 | Number | Trading fees |
| 損益% | Number | P&L percentage |
| 總倉損益% | Number | Total position P&L percentage |
| context_rating | Number | Context quality rating |

### Date/Time Handling
- Supports Excel serial numbers (automatic conversion)
- Accepts various date formats (auto-normalized to YYYY-MM-DD)
- Time format: HH:MM (24-hour)

## 🔧 Development

### Project Structure

```javascript
// Global state
let rawTrades = [];        // All trades from Excel
let filteredTrades = [];   // After applying filters
let filters = [];          // Active filter conditions

// Key functions
init()                     // Application initialization
renderTradesFromRows()     // Populate table from Excel data
applyFilters()             // Execute filter logic
renderAnalytics()          // Update analytics dashboard
renderCharts()             // Render all chart tiles
```

### Adding New Metrics

1. **Edit `js/dashboard_registry.js`**:
   ```javascript
   export const METRIC_REGISTRY = [
     // Add your metric
     { 
       id: 'my_metric', 
       category: 'Custom', 
       label: 'My Custom Metric', 
       format: 'num' 
     }
   ];
   ```

2. **Implement calculation in `index.html`**:
   ```javascript
   // In ANALYTICS_GET function or similar
   case 'my_metric': 
     return { v: myCalculation(trades), kind: 'num' };
   ```

### Adding New Charts

1. **Add to `CHART_REGISTRY`**:
   ```javascript
   { 
     id: 'my_chart', 
     category: 'Custom', 
     label: 'My Chart' 
   }
   ```

2. **Implement renderer**:
   ```javascript
   function renderChartToDiv(el, type, cfg, bigMode) {
     // ...existing charts...
     if (type === 'my_chart') {
       // Your ECharts configuration
       const option = { /* ... */ };
       chart.setOption(option);
     }
   }
   ```

### Building for Distribution

To package as a standalone app:

```bash
# Install electron-builder
npm install --save-dev electron-builder

# Add to package.json
{
  "build": {
    "appId": "com.yourcompany.tradingjournaljournal",
    "productName": "Trading Journal",
    "win": {
      "target": "nsis"
    }
  }
}

# Build
npm run dist
```

## 📝 Notes

- **Language**: UI is primarily in Traditional Chinese (繁體中文)
- **Version**: v0.6 (as indicated in top bar)
- **Browser compatibility**: Electron (Chromium-based)
- **Data persistence**: All changes must be manually saved via "Save Table" button
- **Filter behavior**: Filters use AND logic between rules, OR logic within multi-select options

## 🐛 Troubleshooting

### "desktopExcel API not found"
- Ensure `preload.js` is correctly loaded in `main.js`
- Check `contextBridge` is properly configured

### Excel file not loading
- Verify `data/trading_journal.xlsx` exists
- Check file permissions
- Ensure xlsx package is installed (`npm install xlsx`)

### Charts not rendering
- Confirm ECharts CDN is accessible
- Check browser console for JavaScript errors
- Verify filteredTrades has data

### Filters not working
- Ensure column names match `FIELD_META` definitions
- Check filter operator is valid for column type
- Verify `applyFilters()` is called after filter changes

## 🤝 Contributing

This is a personal trading journal tool. Feel free to fork and customize for your own needs.

## 📄 License

Private/Personal Use

---

**Built with ❤️ for systematic traders**
