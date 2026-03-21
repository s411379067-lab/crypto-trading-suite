// js/dashboard_builder.js
import { getMetricsByCategory, METRIC_REGISTRY } from './dashboard_registry.js';
import { getChartsByCategory, CHART_REGISTRY } from './dashboard_registry.js';
function readMetricFromGlobal(metricId) {
  // 優先用統一計算函式（推薦）
  if (typeof window.ANALYTICS_GET === 'function') {
    const out = window.ANALYTICS_GET(metricId);
    // out: { v, kind }
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
function getMetricValue(metricId) {
  const v = readMetricFromGlobal(metricId);
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
  value.textContent = getMetricValue(initialMetricId);

  const sub = document.createElement('div');
  sub.className = 'muted';
  sub.style.marginTop = '6px';
  sub.textContent = 'Analytics widget (MVP)';

  body.appendChild(value);
  body.appendChild(sub);

  tile.appendChild(header);
  tile.appendChild(body);

  // events
  delBtn.addEventListener('click', () => tile.remove());

  metricSel.addEventListener('change', () => {
    const id = metricSel.value;
    title.textContent = metricLabel(id);
    value.textContent = getMetricValue(id);
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
function renderChartInto(chartId, hostEl) {
  const map = window.CUSTOM_CHART_RENDERERS || {};
  const fn = map[chartId] || map.__default;

  hostEl.innerHTML = '';
  if (typeof fn !== 'function') {
    hostEl.innerHTML = `<div class="muted">No renderer for: ${chartId}</div>`;
    return;
  }

  // 如果用 __default，就把 chartId 傳進去
  if (fn === map.__default) fn(hostEl, chartId);
  else fn(hostEl);
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

//   left.appendChild(title);
  left.appendChild(chartSel);

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

  const draw = () => renderChartInto(chartSel.value, host);

  // events
  delBtn.addEventListener('click', () => tile.remove());
  refreshBtn.addEventListener('click', draw);
  chartSel.addEventListener('change', () => {
    title.textContent = chartLabel(chartSel.value);
    draw();
  });

  // 初次繪製
  draw();

  return tile;
}
function refreshCustomizeTiles() {
  const grid = document.getElementById('customGrid');
  if (!grid) return;

  // 1) refresh analytics tiles
  grid.querySelectorAll('.tile').forEach(tile => {
    const metricSel = tile.querySelector('select.miniSel'); // 你的 metrics/chart 下拉都用 miniSel
    const valueEl = tile.querySelector('.metricValue');
    if (metricSel && valueEl) {
      // analytics tile
      valueEl.textContent = getMetricValue(metricSel.value);
    }
  });

  // 2) refresh chart tiles
  grid.querySelectorAll('.tile').forEach(tile => {
    const host = tile.querySelector('.chartHost');
    if (!host) return;

    const chartSel = tile.querySelector('select.miniSel');
    if (!chartSel) return;

    // 重新畫圖：走你現有的 renderChartInto()
    renderChartInto(chartSel.value, host);
  });
}

export function initCustomizeAddAnalytics() {
  const btnAdd = document.getElementById('btnCusAdd');
  const grid = document.getElementById('customGrid');
  const metricPicker = document.getElementById('cusMetricTest'); // 你現在的 metrics 下拉 id
  const btnRefresh = document.getElementById('btnCusRefresh');

  if (!btnAdd || !grid || !metricPicker) return;

  btnAdd.addEventListener('click', () => {
    const metricId = metricPicker.value || 'winrate';
    const tile = createAnalyticsTile(metricId);
    grid.appendChild(tile);
  });
  btnRefresh?.addEventListener('click', refreshCustomizeTiles);
}
function initCustomizeAddChart() {
  const btnAddChart = document.getElementById('btnCusAddChart');
  const grid = document.getElementById('customGrid');
  const chartPicker = document.getElementById('cusChartTest');
  if (!btnAddChart || !grid || !chartPicker) return;

  btnAddChart.addEventListener('click', () => {
    const chartId = chartPicker.value || 'equity_curve';
    const tile = createChartTile(chartId);
    grid.appendChild(tile);
  });
}



// module 載入後自動初始化（不影響其他頁）
initCustomizeAddAnalytics();
initCustomizeAddChart();