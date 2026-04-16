// js/dashboard_builder.js
import { getMetricsByCategory, METRIC_REGISTRY } from './dashboard_registry.js';
import { getChartsByCategory, CHART_REGISTRY } from './dashboard_registry.js';

const CUSTOM_COMPARE_METRICS = ['avg', 'winrate', 'pf', 'sum', 'count'];

function getCustomizeMetricCfg() {
  return {
    base: document.getElementById('cusMetricBase')?.value || 'pnl',
    feeMode: document.getElementById('cusMetricFee')?.value || 'net'
  };
}

function getCustomizeChartBaseCfg() {
  return {
    base: document.getElementById('cusChartBase')?.value || 'rr',
    fee: document.getElementById('cusChartFee')?.value || 'net',
    roll: Number(document.getElementById('cusChartRoll')?.value || '50'),
    bucket: Number(document.getElementById('cusChartBucket')?.value || '30'),
    timeDim: document.getElementById('cusChartTimeDim')?.value || 'day',
    compareVisible: Number(document.getElementById('cusChartCompareVisible')?.value || '12')
  };
}

function readMetricFromGlobal(metricId, metricCfg) {
  if (typeof window.CUSTOMIZE_ANALYTICS_GET === 'function') {
    return window.CUSTOMIZE_ANALYTICS_GET(metricId, metricCfg);
  }

  if (typeof window.ANALYTICS_GET === 'function') {
    const out = window.ANALYTICS_GET(metricId);
    return out?.v ?? null;
  }

  // fallback：舊的 mapping 物件
  const src = window.ANALYTICS || null;
  if (!src) return null;
  return src[metricId] ?? null;
}

function formatMetricValue(metricId, v) {
  if (v == null || Number.isNaN(v)) return '—';

  // 基本格式：你之後可以依 registry 的 format 做 % / 小數位 / 貨幣 / R
  if (metricId === 'winrate' || metricId === 'mdd') return (v * 100).toFixed(2) + '%';
  return (typeof v === 'number') ? v.toFixed(2) : String(v);
}

function buildMetricSelect(selectedId) {
  const sel = document.createElement('select');
  sel.className = 'miniSel';

  const groups = getMetricsByCategory();
  for (const [cat, items] of Object.entries(groups)) {
    const og = document.createElement('optgroup');
    og.label = cat;
    for (const m of items) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.label;
      if (m.id === selectedId) opt.selected = true;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
  return sel;
}

function metricLabel(metricId) {
  return (METRIC_REGISTRY.find(m => m.id === metricId)?.label) || metricId;
}

// 先做最小可用版：顯示 placeholder（之後再接你真正的 metrics 計算）
function getMetricValue(metricId, metricCfg) {
  const v = readMetricFromGlobal(metricId, metricCfg);
  return formatMetricValue(metricId, v);
}

function createAnalyticsTile(initialMetricId) {
  const tile = document.createElement('div');
  tile.className = 'tile';

  const header = document.createElement('div');
  header.className = 'tileHeader';

  const left = document.createElement('div');
  left.className = 'left';

  const title = document.createElement('strong');
  title.textContent = metricLabel(initialMetricId);

  const metricSel = buildMetricSelect(initialMetricId);

//   left.appendChild(title);
  left.appendChild(metricSel);

  const right = document.createElement('div');
  right.className = 'right';

  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'miniBtn';
  delBtn.textContent = 'Delete';

  right.appendChild(delBtn);

  header.appendChild(left);
  header.appendChild(right);

  const body = document.createElement('div');
  body.className = 'tileBody';

  const value = document.createElement('div');
  value.className = 'metricValue';
  value.style.fontSize = '28px';
  value.style.fontWeight = '800';
  const metricCfg = getCustomizeMetricCfg();
  value.textContent = getMetricValue(initialMetricId, metricCfg);

  const sub = document.createElement('div');
  sub.className = 'muted';
  sub.style.marginTop = '6px';
  sub.textContent = `base=${metricCfg.base} • fee=${metricCfg.feeMode}`;

  body.appendChild(value);
  body.appendChild(sub);

  tile.appendChild(header);
  tile.appendChild(body);

  // events
  delBtn.addEventListener('click', () => tile.remove());

  metricSel.addEventListener('change', () => {
    const cfg = getCustomizeMetricCfg();
    const id = metricSel.value;
    title.textContent = metricLabel(id);
    value.textContent = getMetricValue(id, cfg);
    sub.textContent = `base=${cfg.base} • fee=${cfg.feeMode}`;
  });

  return tile;
}
function buildChartSelect(selectedId) {
  const sel = document.createElement('select');
  sel.className = 'miniSel';

  const groups = getChartsByCategory();
  for (const [cat, items] of Object.entries(groups)) {
    const og = document.createElement('optgroup');
    og.label = cat;
    for (const c of items) {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.label;
      if (c.id === selectedId) opt.selected = true;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
  return sel;
}

function chartLabel(chartId) {
  return (CHART_REGISTRY.find(c => c.id === chartId)?.label) || chartId;
}
function renderChartInto(chartId, hostEl, cfg) {
  const map = window.CUSTOM_CHART_RENDERERS || {};
  const fn = map[chartId] || map.__default;

  hostEl.innerHTML = '';
  if (typeof fn !== 'function') {
    hostEl.innerHTML = `<div class="muted">No renderer for: ${chartId}</div>`;
    return;
  }

  // 如果用 __default，就把 chartId 傳進去
  if (fn === map.__default) fn(hostEl, chartId, cfg);
  else fn(hostEl);
}

function getCustomizeGroupOptions() {
  const sel = document.getElementById('ana_group');
  if (!sel) return [];
  return Array.from(sel.options || []).map(o => ({ value: o.value, label: o.textContent || o.value }));
}

function buildCompareFieldSelect(selectedField) {
  const sel = document.createElement('select');
  sel.className = 'miniSel';
  const opts = getCustomizeGroupOptions();
  opts.forEach(x => {
    const o = document.createElement('option');
    o.value = x.value;
    o.textContent = x.label;
    sel.appendChild(o);
  });
  if (opts.length > 0) {
    sel.value = opts.some(x => x.value === selectedField) ? selectedField : opts[0].value;
  }
  return sel;
}

function buildCompareMetricSelect(selectedMetric) {
  const sel = document.createElement('select');
  sel.className = 'miniSel';
  CUSTOM_COMPARE_METRICS.forEach(m => {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = m;
    sel.appendChild(o);
  });
  sel.value = CUSTOM_COMPARE_METRICS.includes(selectedMetric) ? selectedMetric : 'avg';
  return sel;
}

function createChartTile(initialChartId) {
  const tile = document.createElement('div');
  tile.className = 'tile';

  const header = document.createElement('div');
  header.className = 'tileHeader';

  const left = document.createElement('div');
  left.className = 'left';

  const title = document.createElement('strong');
  title.textContent = chartLabel(initialChartId);

  const chartSel = buildChartSelect(initialChartId);
  const compareFieldSel = buildCompareFieldSelect('strategy');
  const compareMetricSel = buildCompareMetricSelect('avg');

//   left.appendChild(title);
  left.appendChild(chartSel);
  left.appendChild(compareFieldSel);
  left.appendChild(compareMetricSel);

  const right = document.createElement('div');
  right.className = 'right';

  const refreshBtn = document.createElement('button');
  refreshBtn.type = 'button';
  refreshBtn.className = 'miniBtn';
  refreshBtn.textContent = 'Refresh';

  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'miniBtn';
  delBtn.textContent = 'Delete';

//   right.appendChild(refreshBtn);
  right.appendChild(delBtn);

  header.appendChild(left);
  header.appendChild(right);

  const body = document.createElement('div');
  body.className = 'tileBody';

  const host = document.createElement('div');
  host.className = 'chartHost';
  host.id = 'cus_chart_' + Math.random().toString(36).slice(2); // NEW：固定 id
  host.className = 'chartHost';
  host.style.height = '260px'; // 先固定高度，之後做拖拉/resize 再改
  body.appendChild(host);

  tile.appendChild(header);
  tile.appendChild(body);

  const toggleCompareSelectors = () => {
    const show = chartSel.value === 'by_group_metric';
    compareFieldSel.style.display = show ? 'inline-block' : 'none';
    compareMetricSel.style.display = show ? 'inline-block' : 'none';
  };

  const draw = () => {
    const cfg = {
      ...getCustomizeChartBaseCfg(),
      ...(chartSel.value === 'by_group_metric'
        ? { compareField: compareFieldSel.value, compareMetric: compareMetricSel.value }
        : {})
    };
    renderChartInto(chartSel.value, host, cfg);
  };

  // events
  delBtn.addEventListener('click', () => tile.remove());
  refreshBtn.addEventListener('click', draw);
  chartSel.addEventListener('change', () => {
    title.textContent = chartLabel(chartSel.value);
    toggleCompareSelectors();
    draw();
  });
  compareFieldSel.addEventListener('change', draw);
  compareMetricSel.addEventListener('change', draw);
  toggleCompareSelectors();
  if (chartSel.value === 'by_group_metric') {
    draw();
  });

  // 初次繪製
  draw();

  return tile;
}
function refreshCustomizeTiles() {
  const grid = document.getElementById('customGrid');
  if (!grid) return;

  const metricCfg = getCustomizeMetricCfg();

  // 1) refresh analytics tiles
  grid.querySelectorAll('.tile').forEach(tile => {
    const metricSel = tile.querySelector('select.miniSel'); // 你的 metrics/chart 下拉都用 miniSel
    const valueEl = tile.querySelector('.metricValue');
    const subEl = tile.querySelector('.sub');
    if (metricSel && valueEl) {
      // analytics tile
      valueEl.textContent = getMetricValue(metricSel.value, metricCfg);
      if (subEl) subEl.textContent = `base=${metricCfg.base} • fee=${metricCfg.feeMode}`;
    }
  });

  // 2) refresh chart tiles
  grid.querySelectorAll('.tile').forEach(tile => {
    const host = tile.querySelector('.chartHost');
    if (!host) return;

    const chartSel = tile.querySelector('select.miniSel');
    if (!chartSel) return;

    const extraSelectors = tile.querySelectorAll('select.miniSel');
    let compareField = 'strategy';
    let compareMetric = 'avg';
    if (extraSelectors.length >= 3) {
      compareField = extraSelectors[1].value || compareField;
      compareMetric = extraSelectors[2].value || compareMetric;
    }

    const cfg = {
      ...getCustomizeChartBaseCfg(),
      ...(chartSel.value === 'by_group_metric' ? { compareField, compareMetric } : {})
    };

    // 重新畫圖：走你現有的 renderChartInto()
    renderChartInto(chartSel.value, host, cfg);
  });
}

function addCustomizeMetricTile() {
  const grid = document.getElementById('customGrid');
  const metricPicker = document.getElementById('cusMetricTest');
  if (!grid || !metricPicker) return;

  const metricId = metricPicker.value || 'winrate';
  const tile = createAnalyticsTile(metricId);
  grid.appendChild(tile);
}

function addCustomizeChartTile() {
  const grid = document.getElementById('customGrid');
  const chartPicker = document.getElementById('cusChartTest');
  if (!grid || !chartPicker) return;

  const chartId = chartPicker.value || 'equity_curve';
  const tile = createChartTile(chartId);
  grid.appendChild(tile);
}

export function initCustomizeAddAnalytics() {
  const btnAdd = document.getElementById('btnCusAdd');
  const grid = document.getElementById('customGrid');
  const metricPicker = document.getElementById('cusMetricTest'); // 你現在的 metrics 下拉 id
  const btnRefresh = document.getElementById('btnCusRefresh');
  const metricBase = document.getElementById('cusMetricBase');
  const metricFee = document.getElementById('cusMetricFee');

  if (!btnAdd || !grid || !metricPicker) return;
  if (btnAdd.dataset.bound === '1') return;
  btnAdd.dataset.bound = '1';

  btnAdd.addEventListener('click', addCustomizeMetricTile);
  btnRefresh?.addEventListener('click', refreshCustomizeTiles);
  metricBase?.addEventListener('change', refreshCustomizeTiles);
  metricFee?.addEventListener('change', refreshCustomizeTiles);
}
function initCustomizeAddChart() {
  const btnAddChart = document.getElementById('btnCusAddChart');
  const grid = document.getElementById('customGrid');
  const chartPicker = document.getElementById('cusChartTest');
  const chartBase = document.getElementById('cusChartBase');
  const chartFee = document.getElementById('cusChartFee');
  const chartRoll = document.getElementById('cusChartRoll');
  const chartBucket = document.getElementById('cusChartBucket');
  const chartTimeDim = document.getElementById('cusChartTimeDim');
  const chartCompareVisible = document.getElementById('cusChartCompareVisible');
  if (!btnAddChart || !grid || !chartPicker) return;
  if (btnAddChart.dataset.bound === '1') return;
  btnAddChart.dataset.bound = '1';

  btnAddChart.addEventListener('click', addCustomizeChartTile);

  chartBase?.addEventListener('change', refreshCustomizeTiles);
  chartFee?.addEventListener('change', refreshCustomizeTiles);
  chartRoll?.addEventListener('change', refreshCustomizeTiles);
  chartBucket?.addEventListener('change', refreshCustomizeTiles);
  chartTimeDim?.addEventListener('change', refreshCustomizeTiles);
  chartCompareVisible?.addEventListener('change', refreshCustomizeTiles);
}

function bindCustomizeDelegatedEvents() {
  if (window.__customizeBuilderDelegatedBound) return;
  window.__customizeBuilderDelegatedBound = true;

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    if (target.closest('#btnCusAdd')) {
      event.preventDefault();
      addCustomizeMetricTile();
      return;
    }

    if (target.closest('#btnCusAddChart')) {
      event.preventDefault();
      addCustomizeChartTile();
      return;
    }

    if (target.closest('#btnCusRefresh')) {
      event.preventDefault();
      refreshCustomizeTiles();
    }
  });

  document.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) return;

    const refreshIds = new Set([
      'cusMetricBase', 'cusMetricFee',
      'cusChartBase', 'cusChartFee', 'cusChartRoll',
      'cusChartBucket', 'cusChartTimeDim', 'cusChartCompareVisible'
    ]);
    if (refreshIds.has(target.id)) {
      refreshCustomizeTiles();
    }
  });
}



function initCustomizeBuilderWhenReady() {
  bindCustomizeDelegatedEvents();
  initCustomizeAddAnalytics();
  initCustomizeAddChart();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCustomizeBuilderWhenReady, { once: true });
} else {
  initCustomizeBuilderWhenReady();
}