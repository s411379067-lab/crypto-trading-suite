// js/dashboard_registry.js

/* =========================
 * Analytics (單值指標)
 * ========================= */
export const METRIC_REGISTRY = [
  // 損益總覽
  { id:'total_fully_closed', category:'損益總覽', label:'Total (Profit + Loss) – Fully Closed', format:'num' },
  { id:'total_realized',     category:'損益總覽', label:'Total (Profit + Loss) – Realized',     format:'num' },
  { id:'profit_only',        category:'損益總覽', label:'Only Profit PnL',                      format:'num' },
  { id:'loss_only',          category:'損益總覽', label:'Only Loss PnL',                        format:'num' },

  // 單筆/期間平均
  { id:'avg_trade', category:'單筆/期間平均', label:'Per Trade – Average PnL', format:'num' },
  { id:'avg_day',   category:'單筆/期間平均', label:'Per Day – Average PnL',   format:'num' },
  { id:'avg_month', category:'單筆/期間平均', label:'Per Month – Average PnL', format:'num' },
  { id:'avg_year',  category:'單筆/期間平均', label:'Per Year – Average PnL',  format:'num' },

  // 平均贏損
  { id:'avg_win_trade', category:'平均贏損', label:'Per Trade – Average Winning PnL', format:'num' },
  { id:'avg_loss_trade',category:'平均贏損', label:'Per Trade – Average Losing PnL',  format:'num' },
  { id:'avg_win_day',   category:'平均贏損', label:'Per Day – Average Winning PnL',   format:'num' },
  { id:'avg_loss_day',  category:'平均贏損', label:'Per Day – Average Losing PnL',    format:'num' },
  { id:'avg_win_month', category:'平均贏損', label:'Per Month – Average Winning PnL', format:'num' },
  { id:'avg_loss_month',category:'平均贏損', label:'Per Month – Average Losing PnL',  format:'num' },
  { id:'avg_win_year',  category:'平均贏損', label:'Per Year – Average Winning PnL',  format:'num' },
  { id:'avg_loss_year', category:'平均贏損', label:'Per Year – Average Losing PnL',   format:'num' },

  // 極值損益
  { id:'best_trade',  category:'極值損益', label:'Per Trade – Best Winning PnL',    format:'num' },
  { id:'worst_trade', category:'極值損益', label:'Per Trade – Worst Losing PnL',    format:'num' },
  { id:'best_day',    category:'極值損益', label:'Per Day – Best Winning PnL',      format:'num' },
  { id:'worst_day',   category:'極值損益', label:'Per Day – Worst Losing PnL',      format:'num' },
  { id:'best_month',  category:'極值損益', label:'Per Month – Best Winning PnL',    format:'num' },
  { id:'worst_month', category:'極值損益', label:'Per Month – Worst Losing PnL',    format:'num' },
  { id:'best_year',   category:'極值損益', label:'Per Year – Best Winning PnL',     format:'num' },
  { id:'worst_year',  category:'極值損益', label:'Per Year – Worst Losing PnL',     format:'num' },

  // Drawdown
  { id:'dd_current', category:'Drawdown', label:'Drawdown (current)',        format:'num' },
  { id:'mdd',        category:'Drawdown', label:'Max Drawdown',              format:'pct' }, // calcAnaMetric 回傳 kind:num，但你顯示時可當 pct
  { id:'avg_dd',     category:'Drawdown', label:'Average Drawdown (abs)',     format:'num' },
  { id:'ulcer',      category:'Drawdown', label:'Ulcer Index',               format:'num' },

  // 風控/分布
  { id:'winrate',        category:'風控/分布', label:'Win rate',                format:'pct' },
  { id:'pf',             category:'風控/分布', label:'Profit Factor',            format:'num' },
  { id:'payoff',         category:'風控/分布', label:'Payoff',                   format:'num' },
  { id:'sharpe',         category:'風控/分布', label:'Sharpe (trade)',            format:'num' },
  { id:'sortino',        category:'風控/分布', label:'Sortino',                  format:'num' },
  { id:'calmar',         category:'風控/分布', label:'Calmar',                   format:'num' },
  { id:'sd',             category:'風控/分布', label:'Std dev',                  format:'num' },
  { id:'sk',             category:'風控/分布', label:'Skewness',                 format:'num' },
  { id:'ku',             category:'風控/分布', label:'Kurtosis (excess)',         format:'num' },
  { id:'tail',           category:'風控/分布', label:'Tail Ratio (p=0.95)',       format:'num' },
  { id:'max_consec_loss',category:'風控/分布', label:'Max consecutive loss',      format:'int' },
  { id:'max_consec_win', category:'風控/分布', label:'Max consecutive win',       format:'int' },

  // 帳戶數值(僅金額)
  { id:'unrealized',    category:'帳戶數值(僅金額)', label:'Unrealized PnL',     format:'money' },
  { id:'account_value', category:'帳戶數值(僅金額)', label:'Total Account Value',format:'money' }
];

/* =========================
 * Charts（圖表）
 * ========================= */
export const CHART_REGISTRY = [
  { id:'equity',           category:'績效與穩定度', label:'R-based Equity Curve (累積)' },
  { id:'underwater',       category:'績效與穩定度', label:'Underwater (Drawdown)' },
  { id:'trade_series',     category:'績效與穩定度', label:'單筆時間序列' },
  { id:'hist',             category:'分佈',         label:'Histogram' },
  { id:'ecdf',             category:'分佈',         label:'Cumulative Distribution (ECDF)' },
  { id:'roll_avg',         category:'績效與穩定度', label:'Rolling Avg' },
  { id:'roll_win',         category:'績效與穩定度', label:'Rolling Win Rate' },
  { id:'roll_pf',          category:'績效與穩定度', label:'Rolling Profit Factor' },
  { id:'tod_avg',          category:'時間維度',     label:'Time-of-Day Avg (by entry time)' },
  { id:'tod_count',        category:'時間維度',     label:'Time-of-Day Trade Count' },
  { id:'tod_sum',          category:'時間維度',     label:'Time-of-Day Total' },
  { id:'by_strategy_avg',  category:'分類比較',     label:'策略 × Avg' },
  { id:'by_structure_avg', category:'分類比較',     label:'結構 × Avg' },
  { id:'by_context_avg',   category:'分類比較',     label:'Context Rating × Avg' },
  { id:'by_strategy_wr',   category:'分類比較',     label:'策略 × 勝率' },
  { id:'by_strategy_pf',   category:'分類比較',     label:'策略 × Profit Factor' }
];

/* =========================
 * Helper（給下拉用）
 * ========================= */
export function getMetricsByCategory() {
  return groupByCategory(METRIC_REGISTRY);
}

export function getChartsByCategory() {
  return groupByCategory(CHART_REGISTRY);
}

function groupByCategory(list) {
  return list.reduce((acc, item) => {
    acc[item.category] ||= [];
    acc[item.category].push(item);
    return acc;
  }, {});
}
