/**
 * GRID-UP TRANSFORMER EXPLORER - FRONTEND APPLICATION
 * Strict 4-Color System: #0D1117 (C1), #FFFFFF (C2), #38BDF8 (C3), #F59E0B (C4)
 */

// Application State
const state = {
  filters: {
    il: 'All',
    ilce: 'All',
    bolge: 'All',
    status: 'All',
    sortBy: 'train_mean',
    sortOrder: 'desc',
    search: '',
    limit: 50,
    offset: 0
  },
  filterMeta: null,
  activeTanim: null,
  activeTrafoData: null,
  selectedSubmissions: ['v33', 'v24', 'v22'],
  activeTab: 'timeline', // 'timeline', 'yoy', 'monthly', 'district'
  timelineRange: 'all', // 'all', 'last6m', 'forecast'
  districtViewMode: 'total', // 'total', 'avg', 'seen_total', 'seen_avg'
  showTrafoCount: true,
  showTemp: true,
  chartInstance: null,
  trafoListItems: []
};

// Date format helper: converts 'YYYY-MM-DD' to 'DD-MM (dow)' e.g. '2026-08-24' -> '24-08 (mon)'
function formatDayDow(dateStr) {
  if (!dateStr) return '';
  const str = String(dateStr);
  const parts = str.split('-');
  if (parts.length === 3) {
    const y = parseInt(parts[0], 10);
    const m = parseInt(parts[1], 10) - 1;
    const d = parseInt(parts[2], 10);
    const dt = new Date(Date.UTC(y, m, d));
    const dows = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'];
    const dow = dows[dt.getUTCDay()];
    return `${parts[2]}-${parts[1]} (${dow})`;
  }
  return str;
}

// Full date with DOW helper: converts 'YYYY-MM-DD' to 'DD-MM-YYYY (dow)' e.g. '2026-08-24' -> '24-08-2026 (mon)'
function formatDateFullDow(dateStr) {
  if (!dateStr) return '';
  const str = String(dateStr);
  const parts = str.split('-');
  if (parts.length === 3) {
    const y = parseInt(parts[0], 10);
    const m = parseInt(parts[1], 10) - 1;
    const d = parseInt(parts[2], 10);
    const dt = new Date(Date.UTC(y, m, d));
    const dows = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'];
    const dow = dows[dt.getUTCDay()];
    return `${parts[2]}-${parts[1]}-${parts[0]} (${dow})`;
  }
  return str;
}

// Styling constants for Chart.js (Industrial Telemetry Theme)
const PALETTE = {
  c1_bg: '#08090D',
  c2_text: '#FFFFFF',
  c2_dim: '#94A3B8',
  c2_grid: 'rgba(255, 255, 255, 0.08)',
  c3_hist: '#00E5FF',
  c3_fill: 'rgba(0, 229, 255, 0.12)',
  c4_pred: '#FFB800',
  c4_fill: 'rgba(255, 184, 0, 0.12)',
  temp_line: '#E2E8F0',
  temp_hist: 'rgba(0, 229, 255, 0.75)',
  temp_pred: 'rgba(255, 184, 0, 0.75)',
  font_mono: "'JetBrains Mono', monospace"
};

// Distinct dash patterns for submission versions
const SUB_LINE_STYLES = {
  v33: { dash: [], width: 3, label: 'v33 (Active PB)' },
  v24: { dash: [6, 2], width: 2, label: 'v24 (Two-Stage Hurdle)' },
  v23: { dash: [8, 3], width: 2, label: 'v23 (Regime+Agro)' },
  v22: { dash: [4, 4], width: 2, label: 'v22 (Coldstart+Wakeup)' },
  v21: { dash: [6, 4], width: 2, label: 'v21 (Wake-up Floor)' },
  v19: { dash: [2, 2], width: 1.5, label: 'v19 (Mask Prior)' },
  v18: { dash: [8, 3, 2, 3], width: 2, label: 'v18 (Coldstart v2)' },
  v17: { dash: [10, 4], width: 1.5, label: 'v17 (Coldstart v1)' },
  v15: { dash: [4, 4], width: 1.5, label: 'v15 (Ensemble Blend)' },
  v14: { dash: [12, 3], width: 2, label: 'v14 (Drought Features)' },
  v13: { dash: [3, 3], width: 1.5, label: 'v13 (Final Base)' },
  v12: { dash: [1, 3], width: 1.5, label: 'v12 (Ext q55)' },
  v11: { dash: [5, 2], width: 1.5, label: 'v11 (q60)' },
  v10: { dash: [7, 3], width: 1.5, label: 'v10 (Multi-seed)' },
  v9:  { dash: [4, 2, 1, 2], width: 1.5, label: 'v9 (Robust Loss)' },
  v7:  { dash: [9, 5], width: 1.5, label: 'v7 (Horizon)' },
  v6:  { dash: [2, 4], width: 1.5, label: 'v6 (Ensemble Base)' }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadFilterMetadata();
  await loadTransformersList();
});

// Event Listeners
function setupEventListeners() {
  // Search input debounce
  let searchTimeout = null;
  document.getElementById('searchInput').addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      state.filters.search = e.target.value;
      state.filters.offset = 0;
      loadTransformersList();
    }, 250);
  });

  // Filter dropdowns
  document.getElementById('ilSelect').addEventListener('change', (e) => {
    state.filters.il = e.target.value;
    state.filters.ilce = 'All';
    updateIlceDropdown();
    state.filters.offset = 0;
    loadTransformersList();
  });

  document.getElementById('ilceSelect').addEventListener('change', (e) => {
    state.filters.ilce = e.target.value;
    state.filters.offset = 0;
    loadTransformersList();
    if (state.activeTab === 'district') {
      renderCurrentChart();
    }
  });

  document.getElementById('bolgeSelect').addEventListener('change', (e) => {
    state.filters.bolge = e.target.value;
    state.filters.offset = 0;
    loadTransformersList();
  });

  document.getElementById('sortSelect').addEventListener('change', (e) => {
    const [by, order] = e.target.value.split(':');
    state.filters.sortBy = by;
    state.filters.sortOrder = order;
    state.filters.offset = 0;
    loadTransformersList();
  });

  // Status Filter Buttons
  document.querySelectorAll('.status-buttons .filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.status-buttons .filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.filters.status = btn.dataset.status;
      state.filters.offset = 0;
      loadTransformersList();
    });
  });

  // Quick Trafo Navigation
  document.getElementById('btnRandom').addEventListener('click', pickRandomTransformer);
  document.getElementById('btnPrev').addEventListener('click', () => navigateSibling(-1));
  document.getElementById('btnNext').addEventListener('click', () => navigateSibling(1));

  // Chart Tabs
  document.querySelectorAll('.tab-pill, .chart-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab-pill, .chart-tab').forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      state.activeTab = tab.dataset.tab;
      
      const rangeCtrl = document.getElementById('timelineRangeControls');
      if (rangeCtrl) rangeCtrl.style.display = (state.activeTab === 'timeline') ? 'flex' : 'none';

      const distCtrl = document.getElementById('districtViewControls');
      if (distCtrl) distCtrl.style.display = (state.activeTab === 'district') ? 'flex' : 'none';
      
      renderCurrentChart();
    });
  });

  // Timeline Range Controls
  document.querySelectorAll('#timelineRangeControls .range-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#timelineRangeControls .range-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.timelineRange = btn.dataset.range;
      renderCurrentChart();
    });
  });

  // District View Mode Controls
  document.querySelectorAll('#districtViewControls .range-btn[data-district-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#districtViewControls .range-btn[data-district-view]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.districtViewMode = btn.dataset.districtView;
      renderCurrentChart();
    });
  });

  // District Trafo Count Overlay Toggle
  const toggleTrafoBtn = document.getElementById('toggleTrafoCount');
  if (toggleTrafoBtn) {
    toggleTrafoBtn.addEventListener('click', () => {
      state.showTrafoCount = !state.showTrafoCount;
      toggleTrafoBtn.classList.toggle('active', state.showTrafoCount);
      toggleTrafoBtn.textContent = state.showTrafoCount ? 'TRAFOS: ON' : 'TRAFOS: OFF';
      renderCurrentChart();
    });
  }

  // Universal Temperature Overlay Toggle
  const toggleTempBtn = document.getElementById('toggleTempBtn');
  if (toggleTempBtn) {
    toggleTempBtn.addEventListener('click', () => {
      state.showTemp = !state.showTemp;
      toggleTempBtn.classList.toggle('active', state.showTemp);
      toggleTempBtn.textContent = state.showTemp ? 'TEMP (°C): ON' : 'TEMP (°C): OFF';
      renderCurrentChart();
    });
  }

  // Submission Presets
  const presetBtns = document.querySelectorAll('.preset-pill, .preset-btn');
  document.getElementById('presetActive')?.addEventListener('click', (e) => {
    presetBtns.forEach(b => b.classList.remove('active'));
    e.currentTarget.classList.add('active');
    setSubmissionSelection(['v33']);
  });
  document.getElementById('presetTop3')?.addEventListener('click', (e) => {
    presetBtns.forEach(b => b.classList.remove('active'));
    e.currentTarget.classList.add('active');
    setSubmissionSelection(['v33', 'v24', 'v22']);
  });
  document.getElementById('presetColdTrio')?.addEventListener('click', (e) => {
    presetBtns.forEach(b => b.classList.remove('active'));
    e.currentTarget.classList.add('active');
    setSubmissionSelection(['v33', 'v22', 'v18']);
  });
  document.getElementById('presetAll')?.addEventListener('click', (e) => {
    presetBtns.forEach(b => b.classList.remove('active'));
    e.currentTarget.classList.add('active');
    if (state.filterMeta && state.filterMeta.submissions) {
      const allVers = state.filterMeta.submissions.map(s => s.version);
      setSubmissionSelection(allVers);
    }
  });

  // Data Export Buttons
  document.getElementById('btnExportCsv').addEventListener('click', exportCsv);
  document.getElementById('btnExportJson').addEventListener('click', exportJson);
}

// 1. Load Filter Metadata (İl, İlçe, Submissions, Stats)
async function loadFilterMetadata() {
  try {
    const res = await fetch('/api/filters');
    const data = await res.json();
    state.filterMeta = data;

    // Populate İl Dropdown
    const ilSelect = document.getElementById('ilSelect');
    ilSelect.innerHTML = '<option value="All">Tüm İller (İzmir & Manisa)</option>';
    data.iller.forEach(il => {
      const opt = document.createElement('option');
      opt.value = il;
      opt.textContent = `${il} (${data.ilceler_by_il[il]?.length || 0} İlçe)`;
      ilSelect.appendChild(opt);
    });

    updateIlceDropdown();

    // Populate Bölge Dropdown
    const bolgeSelect = document.getElementById('bolgeSelect');
    bolgeSelect.innerHTML = '<option value="All">Tüm Bölgeler</option>';
    data.bolgeler.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = b;
      bolgeSelect.appendChild(opt);
    });

    // Populate Submission Checkboxes
    const subContainer = document.getElementById('submissionCheckboxes');
    subContainer.innerHTML = '';
    data.submissions.forEach(sub => {
      const label = document.createElement('label');
      label.className = `sub-pill-label ${state.selectedSubmissions.includes(sub.version) ? 'checked' : ''}`;
      label.id = `sub_label_${sub.version}`;
      
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.value = sub.version;
      chk.checked = state.selectedSubmissions.includes(sub.version);
      
      chk.addEventListener('change', () => {
        if (chk.checked) {
          if (!state.selectedSubmissions.includes(sub.version)) {
            state.selectedSubmissions.push(sub.version);
          }
          label.classList.add('checked');
        } else {
          state.selectedSubmissions = state.selectedSubmissions.filter(v => v !== sub.version);
          label.classList.remove('checked');
        }
        if (state.selectedSubmissions.length === 0) {
          // Keep at least one selected
          chk.checked = true;
          state.selectedSubmissions = [sub.version];
          label.classList.add('checked');
        }
        refreshActiveTrafoDetails();
      });

      label.appendChild(chk);
      const isActTag = (sub.version === 'v33' || sub.is_active === 1) ? '★ ' : '';
      label.appendChild(document.createTextNode(` ${isActTag}${sub.version}`));
      label.title = `${sub.description} (${sub.filepath})`;
      subContainer.appendChild(label);
    });

    // Load global stats
    const statsRes = await fetch('/api/stats');
    const stats = await statsRes.json();
    document.getElementById('globalTrafoCount').textContent = `${stats.total_transformers.toLocaleString()} Trafos`;

  } catch (err) {
    console.error('Failed to load filter metadata:', err);
  }
}

// Update İlçe dropdown when İl selection changes
function updateIlceDropdown() {
  const ilceSelect = document.getElementById('ilceSelect');
  ilceSelect.innerHTML = '<option value="All">Tüm İlçeler</option>';
  
  if (!state.filterMeta) return;

  const currentIl = state.filters.il;
  const districts = (currentIl === 'All') 
    ? state.filterMeta.all_ilceler 
    : (state.filterMeta.ilceler_by_il[currentIl] || []);

  districts.forEach(ilce => {
    const opt = document.createElement('option');
    opt.value = ilce;
    opt.textContent = ilce;
    ilceSelect.appendChild(opt);
  });
}

// 2. Load List of Transformers from API
async function loadTransformersList() {
  const listContainer = document.getElementById('trafoList');
  listContainer.innerHTML = '<div class="loading-spinner">Searching transformers...</div>';

  try {
    const params = new URLSearchParams({
      il: state.filters.il,
      ilce: state.filters.ilce,
      bolge: state.filters.bolge,
      status: state.filters.status,
      sort_by: state.filters.sortBy,
      sort_order: state.filters.sortOrder,
      q: state.filters.search,
      limit: state.filters.limit,
      offset: state.filters.offset
    });

    const res = await fetch(`/api/transformers?${params.toString()}`);
    const data = await res.json();
    state.trafoListItems = data.items;

    document.getElementById('trafoCountText').textContent = `Transformers (${data.total_count.toLocaleString()})`;

    if (data.items.length === 0) {
      listContainer.innerHTML = '<div class="loading-spinner">No transformers match criteria.</div>';
      return;
    }

    listContainer.innerHTML = '';
    data.items.forEach(t => {
      const card = document.createElement('div');
      card.className = `trafo-item ${t.tanim === state.activeTanim ? 'active' : ''}`;
      card.dataset.tanim = t.tanim;
      
      const badgeClass = t.status === 'Seen' ? 'seen' : (t.status === 'Cold Start' ? 'cold' : '');
      const badgeText = t.status === 'Seen' ? 'SEEN' : (t.status === 'Cold Start' ? 'COLD' : 'TRAIN');

      const histMeanText = t.train_mean ? `${t.train_mean.toLocaleString()} kWh/d` : 'No Train Data';
      const gucText = t.guc ? `${t.guc} kVA` : '-';

      card.innerHTML = `
        <div class="trafo-item-top">
          <span class="trafo-item-id">${t.tanim}</span>
          <span class="trafo-badge ${badgeClass}">${badgeText}</span>
        </div>
        <div class="trafo-item-bottom">
          <span>${t.ilce || t.il} • ${gucText}</span>
          <span>${histMeanText}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        selectTransformer(t.tanim);
      });

      listContainer.appendChild(card);
    });

    // Auto-select first if none selected or not in current view
    if (!state.activeTanim || !data.items.some(t => t.tanim === state.activeTanim)) {
      selectTransformer(data.items[0].tanim);
    } else {
      updateSelectedCardHighlight();
    }

  } catch (err) {
    console.error('Failed to load transformer list:', err);
    listContainer.innerHTML = '<div class="loading-spinner">Error loading transformers.</div>';
  }
}

// 3. Select a Transformer & Fetch Details
async function selectTransformer(tanim) {
  state.activeTanim = tanim;
  updateSelectedCardHighlight();

  try {
    const subStr = state.selectedSubmissions.join(',');
    const res = await fetch(`/api/transformer/${encodeURIComponent(tanim)}?submissions=${subStr}`);
    if (!res.ok) throw new Error('Transformer details not found');
    
    const data = await res.json();
    state.activeTrafoData = data;

    renderProfileCard(data);
    renderCurrentChart();
    renderComparisonTable(data);

  } catch (err) {
    console.error('Error fetching transformer details:', err);
  }
}

async function refreshActiveTrafoDetails() {
  if (state.activeTanim) {
    await selectTransformer(state.activeTanim);
  }
}

function updateSelectedCardHighlight() {
  document.querySelectorAll('.trafo-item').forEach(card => {
    if (card.dataset.tanim === state.activeTanim) {
      card.classList.add('active');
    } else {
      card.classList.remove('active');
    }
  });
}

// 4. Render Active Profile Card & Metrics
function renderProfileCard(data) {
  const t = data.transformer;

  document.getElementById('displayTanim').textContent = t.tanim;
  
  // Format geographic breadcrumb route
  const locEl = document.getElementById('displayLocation');
  const ilText = t.il || 'İZMİR';
  const bolgeText = t.bolge || 'METROPOL';
  const ilceText = t.ilce || 'BORNOVA';
  locEl.innerHTML = `
    <span class="route-node">${ilText}</span>
    <span class="route-sep">›</span>
    <span class="route-node">${bolgeText}</span>
    <span class="route-sep">›</span>
    <span class="route-node highlight">${ilceText}</span>
  `;
  
  const statusBadge = document.getElementById('trafoStatusBadge');
  if (t.status === 'Seen') {
    statusBadge.textContent = 'SEEN // TRAIN & TEST';
    statusBadge.style.color = 'var(--signal-emerald)';
  } else if (t.status === 'Cold Start') {
    statusBadge.textContent = 'COLD START // TEST ONLY';
    statusBadge.style.color = 'var(--signal-tangerine)';
  } else {
    statusBadge.textContent = 'TRAIN ONLY // NO TEST';
    statusBadge.style.color = 'var(--signal-cyan)';
  }

  const gucVal = t.guc ? t.guc : 0;
  document.getElementById('displayGuc').textContent = t.guc ? `${t.guc.toLocaleString()} kVA` : 'Unknown';
  document.getElementById('displayTrainDays').textContent = t.train_days ? `${t.train_days} days` : '0 days';

  // Update capacity gauge bar fill
  const gaugeFill = document.getElementById('gaugeMeterFill');
  if (gaugeFill) {
    const fillPct = gucVal > 0 ? Math.min(100, Math.max(12, Math.round((gucVal / 2500) * 100))) : 20;
    gaugeFill.style.width = `${fillPct}%`;
  }

  // Metrics
  document.getElementById('valHistMean').textContent = t.train_mean ? `${t.train_mean.toLocaleString()} kWh/d` : 'N/A';
  document.getElementById('valHistMax').textContent = t.train_max ? `Peak: ${t.train_max.toLocaleString()} kWh` : 'Peak: N/A';

  document.getElementById('valPredMean').textContent = t.pred_mean ? `${t.pred_mean.toLocaleString()} kWh/d` : 'N/A';
  document.getElementById('valPredMax').textContent = t.pred_max ? `Peak: ${t.pred_max.toLocaleString()} kWh` : 'Peak: N/A';

  const predSumMwh = t.pred_sum ? (t.pred_sum / 1000).toFixed(1) : 'N/A';
  document.getElementById('valPredSum').textContent = predSumMwh !== 'N/A' ? `${predSumMwh} MWh` : 'N/A';

  const yoyText = (t.yoy_pct !== null && t.yoy_pct !== undefined) 
    ? `${t.yoy_pct > 0 ? '+' : ''}${t.yoy_pct}%` 
    : 'N/A';
  const yoyEl = document.getElementById('valYoYPct');
  yoyEl.textContent = yoyText;
  yoyEl.style.color = (t.yoy_pct !== null && t.yoy_pct > 0) ? 'var(--signal-amber)' : 'var(--signal-cyan)';
}

// 5. Chart Rendering Engine (Strict 4 Colors)
function renderCurrentChart() {
  if (state.chartInstance) {
    state.chartInstance.destroy();
    state.chartInstance = null;
  }

  const ctx = document.getElementById('mainChart').getContext('2d');
  const customLegend = document.getElementById('customLegend');
  customLegend.innerHTML = '';

  if (state.activeTab === 'timeline') {
    renderTimelineChart(ctx, customLegend);
  } else if (state.activeTab === 'yoy') {
    renderYoYChart(ctx, customLegend);
  } else if (state.activeTab === 'monthly') {
    renderMonthlyChart(ctx, customLegend);
  } else if (state.activeTab === 'district') {
    renderDistrictAggregateChart(ctx, customLegend);
  }
}

// Tab 1: Continuous Timeline Chart
function renderTimelineChart(ctx, legendEl) {
  if (!state.activeTrafoData) return;

  const data = state.activeTrafoData;
  let histPoints = data.historical || [];
  const predictions = data.predictions || {};
  const weather = data.weather || [];

  // Filter based on timelineRange
  if (state.timelineRange === 'last6m') {
    // 6 months before forecast: 2025-10-01 onwards
    histPoints = histPoints.filter(p => p.tarih >= '2025-10-01');
  } else if (state.timelineRange === 'forecast') {
    histPoints = [];
  }

  // Combine dates
  const histDates = histPoints.map(p => p.tarih);
  const subDates = predictions[data.selected_versions[0]]?.map(p => p.tarih) || [];
  const allDates = [...histDates, ...subDates];

  const datasets = [];

  // 1. Historical Dataset (Sky Blue #38BDF8)
  if (histPoints.length > 0) {
    const histDataPoints = allDates.map(dt => {
      const found = histPoints.find(p => p.tarih === dt);
      return found ? found.tuketim : null;
    });

    datasets.push({
      label: 'Historical Actual (Train)',
      data: histDataPoints,
      borderColor: PALETTE.c3_hist,
      backgroundColor: PALETTE.c3_fill,
      borderWidth: 2.5,
      pointRadius: allDates.length > 100 ? 0 : 2,
      pointHoverRadius: 5,
      pointBackgroundColor: PALETTE.c3_hist,
      tension: 0.1,
      yAxisID: 'y',
      fill: true
    });

    addLegendItem(legendEl, 'Historical Actuals (2025-01 -> 2026-03)', PALETTE.c3_hist, []);
  }

  // 2. Submission Datasets (Amber Gold #F59E0B with distinct dashes)
  data.selected_versions.forEach(ver => {
    const subPoints = predictions[ver] || [];
    const style = SUB_LINE_STYLES[ver] || { dash: [4, 4], width: 2, label: ver };

    const predDataPoints = allDates.map(dt => {
      const found = subPoints.find(p => p.tarih === dt);
      return found ? found.tuketim : null;
    });

    datasets.push({
      label: `Submission ${ver}: ${style.label}`,
      data: predDataPoints,
      borderColor: PALETTE.c4_pred,
      borderDash: style.dash,
      borderWidth: style.width,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointBackgroundColor: PALETTE.c4_pred,
      tension: 0.1,
      yAxisID: 'y',
      fill: false
    });

    addLegendItem(legendEl, `Forecast: ${style.label}`, PALETTE.c4_pred, style.dash);
  });

  // 3. District Temperature Dataset on secondary axis y1
  const showTemp = state.showTemp !== false;
  if (showTemp) {
    const wxMap = {};
    weather.forEach(w => { wxMap[w.tarih] = w.temp_mean ?? w.temp; });
    histPoints.forEach(h => { if (h.temp !== undefined && h.temp !== null) wxMap[h.tarih] = h.temp; });
    Object.values(predictions).forEach(arr => {
      (arr || []).forEach(p => { if (p.temp !== undefined && p.temp !== null) wxMap[p.tarih] = p.temp; });
    });

    const tempDataPoints = allDates.map(dt => {
      const wVal = wxMap[dt];
      return wVal !== undefined && wVal !== null ? roundVal(wVal, 1) : null;
    });

    if (tempDataPoints.some(v => v !== null)) {
      datasets.push({
        label: 'Temperature (°C)',
        data: tempDataPoints,
        borderColor: PALETTE.temp_line,
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        borderDash: [3, 2],
        pointRadius: 0,
        pointHoverRadius: 4,
        pointBackgroundColor: PALETTE.c2_text,
        tension: 0.2,
        yAxisID: 'y1',
        fill: false
      });

      addLegendItem(legendEl, 'Temperature (°C) [Right Axis]', PALETTE.temp_line, [3, 2]);
    }
  }

  state.chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: allDates,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: PALETTE.c1_bg,
          titleColor: PALETTE.c2_text,
          bodyColor: PALETTE.c2_text,
          borderColor: PALETTE.c3_hist,
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: function(items) {
              const dt = items[0].label;
              return `Date: ${formatDayDow(dt)} (${dt})`;
            },
            label: function(context) {
              const val = context.parsed.y;
              if (val === null || val === undefined) return null;
              if (context.dataset.yAxisID === 'y1') {
                return ` ${context.dataset.label}: ${val.toFixed(1)} °C`;
              }
              return ` ${context.dataset.label}: ${val.toLocaleString()} kWh`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: PALETTE.c2_grid },
          ticks: {
            color: PALETTE.c2_text,
            maxTicksLimit: 14,
            font: { family: 'monospace', size: 10 },
            callback: function(val, index) {
              const dt = this.getLabelForValue(val);
              return formatDayDow(dt);
            }
          }
        },
        y: {
          position: 'left',
          grid: { color: PALETTE.c2_grid },
          ticks: {
            color: PALETTE.c2_text,
            callback: v => `${v.toLocaleString()} kWh`,
            font: { family: 'monospace', size: 10 }
          },
          title: {
            display: true,
            text: 'Consumption (kWh)',
            color: PALETTE.c2_dim,
            font: { family: 'monospace', size: 10 }
          }
        },
        ...(showTemp ? {
          y1: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: {
              color: PALETTE.temp_line,
              callback: v => `${v}°C`,
              font: { family: 'monospace', size: 10 }
            },
            title: {
              display: true,
              text: 'Temperature (°C)',
              color: PALETTE.temp_line,
              font: { family: 'monospace', size: 10 }
            }
          }
        } : {})
      }
    }
  });
}

// Tab 2: Year-over-Year (YoY) Overlay Chart
function renderYoYChart(ctx, legendEl) {
  if (!state.activeTrafoData) return;
  const data = state.activeTrafoData;
  const yoy = data.yoy || [];

  if (yoy.length === 0) {
    legendEl.innerHTML = '<span style="color:var(--c4)">No aligned summer data for YoY comparison.</span>';
    return;
  }

  const labels = yoy.map(r => r.day_label || formatDayDow(r.pred_date)); // e.g. "15-04 (wed)"
  const datasets = [];

  // Historical 2025 Summer (Sky Blue #38BDF8)
  const histVals = yoy.map(r => r.hist_val);
  datasets.push({
    label: '2025 Historical Summer (Apr - Jul 2025)',
    data: histVals,
    borderColor: PALETTE.c3_hist,
    backgroundColor: PALETTE.c3_fill,
    borderWidth: 2.5,
    pointRadius: 0,
    tension: 0.1,
    yAxisID: 'y',
    fill: true
  });
  addLegendItem(legendEl, 'Historical Summer 2025 (Actual)', PALETTE.c3_hist, []);

  // Submissions 2026 Summer (Amber #F59E0B)
  data.selected_versions.forEach(ver => {
    const style = SUB_LINE_STYLES[ver] || { dash: [4, 4], width: 2, label: ver };
    const vals = yoy.map(r => r[ver]);

    datasets.push({
      label: `2026 Forecast ${ver}: ${style.label}`,
      data: vals,
      borderColor: PALETTE.c4_pred,
      borderDash: style.dash,
      borderWidth: style.width,
      pointRadius: 0,
      tension: 0.1,
      yAxisID: 'y',
      fill: false
    });
    addLegendItem(legendEl, `Forecast Summer 2026 (${ver})`, PALETTE.c4_pred, style.dash);
  });

  // 2025 & 2026 Summer Temperature on y1
  const showTemp = state.showTemp !== false;
  if (showTemp) {
    const histTempVals = yoy.map(r => r.hist_temp !== undefined && r.hist_temp !== null ? roundVal(r.hist_temp, 1) : null);
    if (histTempVals.some(v => v !== null)) {
      datasets.push({
        label: '2025 Summer Temp (°C)',
        data: histTempVals,
        borderColor: PALETTE.temp_hist,
        borderDash: [3, 2],
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.2,
        yAxisID: 'y1',
        fill: false
      });
      addLegendItem(legendEl, 'Summer 2025 Temp (°C)', PALETTE.temp_hist, [3, 2]);
    }

    const predTempVals = yoy.map(r => r.pred_temp !== undefined && r.pred_temp !== null ? roundVal(r.pred_temp, 1) : null);
    if (predTempVals.some(v => v !== null)) {
      datasets.push({
        label: '2026 Summer Temp (°C)',
        data: predTempVals,
        borderColor: PALETTE.temp_pred,
        borderDash: [3, 2],
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.2,
        yAxisID: 'y1',
        fill: false
      });
      addLegendItem(legendEl, 'Summer 2026 Temp (°C)', PALETTE.temp_pred, [3, 2]);
    }
  }

  state.chartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: PALETTE.c1_bg,
          titleColor: PALETTE.c2_text,
          bodyColor: PALETTE.c2_text,
          borderColor: PALETTE.c4_pred,
          borderWidth: 1,
          callbacks: {
            title: function(items) {
              const idx = items[0].dataIndex;
              const y = yoy[idx];
              return `Day: ${y.day_label} [2025: ${y.hist_date} (${y.hist_day_label || ''}) vs 2026: ${y.pred_date}]`;
            },
            label: function(context) {
              const val = context.parsed.y;
              if (val === null || val === undefined) return null;
              if (context.dataset.yAxisID === 'y1') {
                return ` ${context.dataset.label}: ${val.toFixed(1)} °C`;
              }
              return ` ${context.dataset.label}: ${val.toLocaleString()} kWh`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: PALETTE.c2_grid },
          ticks: { color: PALETTE.c2_text, maxTicksLimit: 14, font: { family: 'monospace', size: 10 } }
        },
        y: {
          position: 'left',
          grid: { color: PALETTE.c2_grid },
          ticks: { color: PALETTE.c2_text, callback: v => `${v.toLocaleString()} kWh`, font: { family: 'monospace', size: 10 } },
          title: { display: true, text: 'Consumption (kWh)', color: PALETTE.c2_dim, font: { family: 'monospace', size: 10 } }
        },
        ...(showTemp ? {
          y1: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: PALETTE.temp_line, callback: v => `${v}°C`, font: { family: 'monospace', size: 10 } },
            title: { display: true, text: 'Temperature (°C)', color: PALETTE.temp_line, font: { family: 'monospace', size: 10 } }
          }
        } : {})
      }
    }
  });
}

// Tab 3: Monthly Breakdown Bar Chart
function renderMonthlyChart(ctx, legendEl) {
  if (!state.activeTrafoData) return;
  const data = state.activeTrafoData;
  const monthly = data.monthly || [];

  const labels = monthly.map(m => m.month); // "2025-01", ... "2026-07"
  const datasets = [];

  // Historical Monthly Avg
  const histAvgs = monthly.map(m => m.hist_avg);
  datasets.push({
    label: 'Historical Daily Avg (kWh/day)',
    data: histAvgs,
    backgroundColor: PALETTE.c3_hist,
    borderColor: PALETTE.c3_hist,
    borderWidth: 1,
    yAxisID: 'y'
  });
  addLegendItem(legendEl, 'Historical Monthly Average (kWh/d)', PALETTE.c3_hist, []);

  // Main Submission v22 (or first selected)
  const mainVer = data.selected_versions[0] || 'v22';
  const subAvgs = monthly.map(m => m[`${mainVer}_avg`]);
  datasets.push({
    label: `Forecast Daily Avg (${mainVer})`,
    data: subAvgs,
    backgroundColor: PALETTE.c4_pred,
    borderColor: PALETTE.c4_pred,
    borderWidth: 1,
    yAxisID: 'y'
  });
  addLegendItem(legendEl, `Forecast Monthly Average: ${mainVer} (kWh/d)`, PALETTE.c4_pred, []);

  // Monthly Mean Temperature (°C) Line Overlay on y1
  const showTemp = state.showTemp !== false;
  const tempAvgs = monthly.map(m => m.temp_avg !== undefined && m.temp_avg !== null ? roundVal(m.temp_avg, 1) : null);
  if (showTemp && tempAvgs.some(v => v !== null)) {
    datasets.push({
      label: 'Monthly Mean Temp (°C)',
      type: 'line',
      data: tempAvgs,
      borderColor: PALETTE.c2_text,
      backgroundColor: PALETTE.c2_text,
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.2,
      yAxisID: 'y1',
      fill: false
    });
    addLegendItem(legendEl, 'Monthly Mean Temperature (°C)', PALETTE.c2_text, []);
  }

  state.chartInstance = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: PALETTE.c1_bg,
          titleColor: PALETTE.c2_text,
          bodyColor: PALETTE.c2_text,
          borderColor: PALETTE.c3_hist,
          borderWidth: 1,
          callbacks: {
            label: function(context) {
              const val = context.parsed.y;
              if (val === null || val === undefined) return null;
              if (context.dataset.yAxisID === 'y1') {
                return ` ${context.dataset.label}: ${val.toFixed(1)} °C`;
              }
              return ` ${context.dataset.label}: ${val.toLocaleString()} kWh/d`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: PALETTE.c2_grid },
          ticks: { color: PALETTE.c2_text, font: { family: 'monospace', size: 10 } }
        },
        y: {
          position: 'left',
          grid: { color: PALETTE.c2_grid },
          ticks: { color: PALETTE.c2_text, callback: v => `${v.toLocaleString()} kWh/d`, font: { family: 'monospace', size: 10 } },
          title: { display: true, text: 'Daily Avg (kWh/d)', color: PALETTE.c2_dim, font: { family: 'monospace', size: 10 } }
        },
        ...(showTemp ? {
          y1: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: PALETTE.temp_line, callback: v => `${v}°C`, font: { family: 'monospace', size: 10 } },
            title: { display: true, text: 'Temperature (°C)', color: PALETTE.temp_line, font: { family: 'monospace', size: 10 } }
          }
        } : {})
      }
    }
  });
}

// Tab 4: District Aggregate Chart
async function renderDistrictAggregateChart(ctx, legendEl) {
  const il = state.filters.il !== 'All' ? state.filters.il : (state.activeTrafoData?.transformer.il || 'İZMİR');
  const ilce = state.filters.ilce !== 'All' ? state.filters.ilce : (state.activeTrafoData?.transformer.ilce || 'BORNOVA');

  legendEl.innerHTML = '<span style="color:var(--c3)">Loading district totals for ' + il + ' > ' + ilce + '...</span>';

  try {
    const subStr = state.selectedSubmissions.join(',');
    const res = await fetch(`/api/district?il=${encodeURIComponent(il)}&ilce=${encodeURIComponent(ilce)}&submissions=${subStr}`);
    const data = await res.json();

    legendEl.innerHTML = '';
    const hist = data.historical || [];
    const preds = data.predictions || {};
    const weather = data.weather || [];
    const distInfo = data.district_info || {};

    const allDates = [...hist.map(h => h.tarih), ...(preds[data.selected_versions[0]]?.map(p => p.tarih) || [])];
    const mode = state.districtViewMode || 'total';
    const showTrafoCount = state.showTrafoCount !== false;

    // View mode parameters
    let unitLabel = 'MWh/day';
    let yAxisTitle = 'Total Generation / Consumption (MWh/day)';
    let yTickFormatter = v => `${v.toLocaleString()} MWh`;
    let histLabel = `Historical District Total (${ilce})`;
    let predLabelPrefix = 'Forecast District Total';

    if (mode === 'avg') {
      unitLabel = 'kWh/trafo/day';
      yAxisTitle = 'Daily Avg per Transformer (kWh/trafo/day)';
      yTickFormatter = v => `${v.toLocaleString()} kWh/d`;
      histLabel = `Historical Daily Avg / Trafo (${ilce})`;
      predLabelPrefix = 'Forecast Daily Avg / Trafo';
    } else if (mode === 'seen_total') {
      unitLabel = 'MWh/day';
      yAxisTitle = 'Constant Fleet Total (Seen Only MWh/day)';
      yTickFormatter = v => `${v.toLocaleString()} MWh`;
      histLabel = `Historical Constant Fleet Total (${ilce})`;
      predLabelPrefix = 'Forecast Constant Fleet Total';
    } else if (mode === 'seen_avg') {
      unitLabel = 'kWh/trafo/day';
      yAxisTitle = 'Constant Fleet Avg per Trafo (kWh/trafo/day)';
      yTickFormatter = v => `${v.toLocaleString()} kWh/d`;
      histLabel = `Historical Constant Fleet Avg / Trafo (${ilce})`;
      predLabelPrefix = 'Forecast Constant Fleet Avg / Trafo';
    }

    const datasets = [];

    // Helper to extract value based on mode
    function extractHistVal(h) {
      if (!h) return null;
      if (mode === 'total') return roundVal(h.total_tuketim / 1000, 2);
      if (mode === 'avg') return roundVal(h.avg_tuketim, 1);
      if (mode === 'seen_total') return roundVal((h.seen_total_tuketim ?? h.total_tuketim) / 1000, 2);
      if (mode === 'seen_avg') return roundVal(h.seen_avg_tuketim ?? h.avg_tuketim, 1);
      return null;
    }

    function extractPredVal(p) {
      if (!p) return null;
      if (mode === 'total') return roundVal(p.total_tuketim / 1000, 2);
      if (mode === 'avg') return roundVal(p.avg_tuketim, 1);
      if (mode === 'seen_total') return roundVal((p.seen_total_tuketim ?? p.total_tuketim) / 1000, 2);
      if (mode === 'seen_avg') return roundVal(p.seen_avg_tuketim ?? p.avg_tuketim, 1);
      return null;
    }

    // 1. Historical Dataset
    const histMap = {};
    hist.forEach(h => { histMap[h.tarih] = h; });

    const histData = allDates.map(dt => extractHistVal(histMap[dt]));

    datasets.push({
      label: histLabel,
      data: histData,
      borderColor: PALETTE.c3_hist,
      backgroundColor: PALETTE.c3_fill,
      borderWidth: 2,
      pointRadius: 0,
      yAxisID: 'y',
      fill: true
    });

    const histLegendTrafos = (mode.startsWith('seen')) 
      ? (distInfo.seen_trafo_count ?? distInfo.trafo_count) 
      : (distInfo.trafo_count ?? 'all');
    addLegendItem(legendEl, `${histLabel} (${histLegendTrafos} trafos)`, PALETTE.c3_hist, []);

    // 2. Submission Datasets
    const predMaps = {};
    data.selected_versions.forEach(ver => {
      const subPoints = preds[ver] || [];
      const subMap = {};
      subPoints.forEach(p => { subMap[p.tarih] = p; });
      predMaps[ver] = subMap;

      const style = SUB_LINE_STYLES[ver] || { dash: [4, 4], width: 2, label: ver };
      const predData = allDates.map(dt => extractPredVal(subMap[dt]));

      datasets.push({
        label: `${predLabelPrefix} (${ver})`,
        data: predData,
        borderColor: PALETTE.c4_pred,
        borderDash: style.dash,
        borderWidth: style.width,
        pointRadius: 0,
        yAxisID: 'y',
        fill: false
      });
      addLegendItem(legendEl, `${predLabelPrefix} (${ver})`, PALETTE.c4_pred, style.dash);
    });

    // 3. District Temperature on y1
    const showTemp = state.showTemp !== false;
    if (showTemp) {
      const distWxMap = {};
      weather.forEach(w => { distWxMap[w.tarih] = w.temp_mean ?? w.temp; });
      hist.forEach(h => { if (h.temp !== undefined && h.temp !== null) distWxMap[h.tarih] = h.temp; });
      Object.values(preds).forEach(arr => {
        (arr || []).forEach(p => { if (p.temp !== undefined && p.temp !== null) distWxMap[p.tarih] = p.temp; });
      });

      const distTempData = allDates.map(dt => {
        const wVal = distWxMap[dt];
        return wVal !== undefined && wVal !== null ? roundVal(wVal, 1) : null;
      });

      if (distTempData.some(v => v !== null)) {
        datasets.push({
          label: 'District Temperature (°C)',
          data: distTempData,
          borderColor: PALETTE.temp_line,
          borderDash: [3, 2],
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.2,
          yAxisID: 'y1',
          fill: false
        });
        addLegendItem(legendEl, 'Temperature (°C) [Right Axis]', PALETTE.temp_line, [3, 2]);
      }
    }

    // 4. Active Transformer Count on y2 (if toggled)
    if (showTrafoCount) {
      const trafoCountData = allDates.map(dt => {
        if (histMap[dt]) {
          return mode.startsWith('seen') ? histMap[dt].seen_trafo_count : histMap[dt].trafo_count;
        }
        const firstPred = predMaps[data.selected_versions[0]]?.[dt];
        if (firstPred) {
          return mode.startsWith('seen') ? firstPred.seen_trafo_count : firstPred.trafo_count;
        }
        return null;
      });

      if (trafoCountData.some(v => v !== null)) {
        datasets.push({
          label: mode.startsWith('seen') ? 'Active Seen Transformers' : 'Active Reporting Transformers',
          data: trafoCountData,
          borderColor: 'rgba(255, 255, 255, 0.45)',
          borderDash: [1, 2],
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0,
          yAxisID: 'y2',
          fill: false
        });
        addLegendItem(legendEl, 'Active Trafo Count [Right Axis]', 'rgba(255, 255, 255, 0.5)', [1, 2]);
      }
    }

    // Configure Scales
    const scalesConfig = {
      x: {
        grid: { color: PALETTE.c2_grid },
        ticks: {
          color: PALETTE.c2_text,
          maxTicksLimit: 14,
          font: { family: 'monospace', size: 10 },
          callback: function(val) {
            const dt = this.getLabelForValue(val);
            return formatDayDow(dt);
          }
        }
      },
      y: {
        position: 'left',
        grid: { color: PALETTE.c2_grid },
        ticks: {
          color: PALETTE.c2_text,
          callback: yTickFormatter,
          font: { family: 'monospace', size: 10 }
        },
        title: {
          display: true,
          text: yAxisTitle,
          color: PALETTE.c2_dim,
          font: { family: 'monospace', size: 10 }
        }
      },
      ...(showTemp ? {
        y1: {
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: {
            color: PALETTE.temp_line,
            callback: v => `${v}°C`,
            font: { family: 'monospace', size: 10 }
          },
          title: {
            display: true,
            text: 'Temperature (°C)',
            color: PALETTE.temp_line,
            font: { family: 'monospace', size: 10 }
          }
        }
      } : {})
    };

    if (showTrafoCount) {
      scalesConfig.y2 = {
        position: 'right',
        grid: { drawOnChartArea: false },
        ticks: {
          color: 'rgba(255, 255, 255, 0.45)',
          callback: v => `${Math.round(v)}`,
          font: { family: 'monospace', size: 9 }
        },
        title: {
          display: true,
          text: 'Trafos',
          color: 'rgba(255, 255, 255, 0.45)',
          font: { family: 'monospace', size: 9 }
        }
      };
    }

    state.chartInstance = new Chart(ctx, {
      type: 'line',
      data: { labels: allDates, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: PALETTE.c1_bg,
            titleColor: PALETTE.c2_text,
            bodyColor: PALETTE.c2_text,
            borderColor: PALETTE.c3_hist,
            borderWidth: 1,
            callbacks: {
              title: function(items) {
                const dt = items[0].label;
                const hObj = histMap[dt];
                const pObj = predMaps[data.selected_versions[0]]?.[dt];
                const tCount = hObj ? hObj.trafo_count : (pObj ? pObj.trafo_count : '-');
                const sCount = hObj ? hObj.seen_trafo_count : (pObj ? pObj.seen_trafo_count : '-');
                return `Date: ${formatDayDow(dt)} (${dt}) • ${tCount} trafos active (${sCount} seen)`;
              },
              label: function(context) {
                const val = context.parsed.y;
                if (val === null || val === undefined) return null;
                if (context.dataset.yAxisID === 'y1') {
                  return ` ${context.dataset.label}: ${val.toFixed(1)} °C`;
                }
                if (context.dataset.yAxisID === 'y2') {
                  return ` ${context.dataset.label}: ${Math.round(val)} trafos`;
                }
                return ` ${context.dataset.label}: ${val.toLocaleString()} ${unitLabel}`;
              }
            }
          }
        },
        scales: scalesConfig
      }
    });

  } catch (err) {
    console.error('Error fetching district aggregate:', err);
    legendEl.innerHTML = '<span style="color:var(--c4)">Failed to load district aggregate data.</span>';
  }
}

// 6. Populate Comparison Data Table
function renderComparisonTable(data) {
  const tableHead = document.getElementById('tableHeadRow');
  const tableBody = document.getElementById('tableBody');

  const selectedVers = data.selected_versions || ['v22'];
  
  // Rebuild Table Header with Day of Week and Temperature
  let headHtml = '<th>Date (Day)</th><th>Period</th><th>Temp (°C)</th><th>Historical (kWh)</th>';
  selectedVers.forEach(v => {
    headHtml += `<th>${v} (kWh)</th>`;
  });
  headHtml += '<th>Delta vs Hist (%)</th>';
  tableHead.innerHTML = headHtml;

  // Build daily rows
  const hist = data.historical || [];
  const preds = data.predictions || {};
  const yoyMap = {};
  (data.yoy || []).forEach(y => {
    yoyMap[y.pred_date] = y;
  });

  const subDates = preds[selectedVers[0]]?.map(p => p.tarih) || [];
  const totalRows = subDates.length;

  if (totalRows === 0 && hist.length === 0) {
    tableBody.innerHTML = '<tr><td colspan="10" class="text-center">No daily records available.</td></tr>';
    return;
  }

  // Show last 30 historical days + all forecast days
  let rowsHtml = '';
  const histSlice = hist.slice(-30);
  histSlice.forEach(h => {
    const dayLabel = h.day_label || formatDayDow(h.tarih);
    const tempText = (h.temp !== undefined && h.temp !== null) ? `${Number(h.temp).toFixed(1)}°C` : '-';
    rowsHtml += `
      <tr>
        <td><strong>${dayLabel}</strong> <span style="opacity:0.5; font-size:11px;">(${h.tarih})</span></td>
        <td><span style="color:var(--c3)">Historical</span></td>
        <td style="color:var(--c2-dim)">${tempText}</td>
        <td>${h.tuketim.toLocaleString()}</td>
        ${selectedVers.map(() => '<td>-</td>').join('')}
        <td>-</td>
      </tr>
    `;
  });

  subDates.forEach(dt => {
    const yoyEntry = yoyMap[dt];
    const histCompareVal = yoyEntry ? yoyEntry.hist_val : null;
    const v22Val = preds['v22']?.find(p => p.tarih === dt)?.tuketim;
    const predTemp = yoyEntry?.pred_temp ?? preds[selectedVers[0]]?.find(p => p.tarih === dt)?.temp;
    const tempText = (predTemp !== undefined && predTemp !== null) ? `${Number(predTemp).toFixed(1)}°C` : '-';
    const dayLabel = yoyEntry?.day_label || formatDayDow(dt);

    let deltaPctText = '-';
    if (histCompareVal && v22Val && histCompareVal > 0) {
      const dPct = ((v22Val - histCompareVal) / histCompareVal) * 100;
      deltaPctText = `${dPct > 0 ? '+' : ''}${dPct.toFixed(1)}%`;
    }

    rowsHtml += `
      <tr>
        <td><strong>${dayLabel}</strong> <span style="opacity:0.5; font-size:11px;">(${dt})</span></td>
        <td><span style="color:var(--c4)">Forecast</span></td>
        <td style="color:var(--c2-dim)">${tempText}</td>
        <td>${histCompareVal ? histCompareVal.toLocaleString() : '<span style="opacity:0.5">-</span>'}</td>
        ${selectedVers.map(v => {
          const val = preds[v]?.find(p => p.tarih === dt)?.tuketim;
          return `<td>${val !== undefined && val !== null ? val.toLocaleString() : '-'}</td>`;
        }).join('')}
        <td style="color:${deltaPctText.startsWith('+') ? 'var(--c4)' : 'var(--c3)'}">${deltaPctText}</td>
      </tr>
    `;
  });

  tableBody.innerHTML = rowsHtml;
}

// 7. Navigation & Utility Functions
function pickRandomTransformer() {
  if (state.trafoListItems.length > 0) {
    const randIdx = Math.floor(Math.random() * state.trafoListItems.length);
    selectTransformer(state.trafoListItems[randIdx].tanim);
  }
}

function navigateSibling(dir) {
  if (!state.trafoListItems || state.trafoListItems.length === 0) return;
  const currIdx = state.trafoListItems.findIndex(t => t.tanim === state.activeTanim);
  if (currIdx === -1) {
    selectTransformer(state.trafoListItems[0].tanim);
    return;
  }
  let nextIdx = currIdx + dir;
  if (nextIdx < 0) nextIdx = state.trafoListItems.length - 1;
  if (nextIdx >= state.trafoListItems.length) nextIdx = 0;
  selectTransformer(state.trafoListItems[nextIdx].tanim);
}

function setSubmissionSelection(versions) {
  state.selectedSubmissions = versions;
  if (state.filterMeta && state.filterMeta.submissions) {
    state.filterMeta.submissions.forEach(s => {
      const lbl = document.getElementById(`sub_label_${s.version}`);
      const chk = lbl?.querySelector('input[type="checkbox"]');
      if (chk) {
        chk.checked = versions.includes(s.version);
        if (chk.checked) lbl.classList.add('checked');
        else lbl.classList.remove('checked');
      }
    });
  }
  refreshActiveTrafoDetails();
}

function addLegendItem(container, labelText, color, dash) {
  const item = document.createElement('div');
  item.className = 'legend-item';

  const line = document.createElement('span');
  line.className = 'legend-line';
  line.style.backgroundColor = color;
  if (dash && dash.length > 0) {
    line.style.borderTop = `2px dashed ${color}`;
    line.style.backgroundColor = 'transparent';
  }

  const text = document.createElement('span');
  text.textContent = labelText;
  text.style.color = color;

  item.appendChild(line);
  item.appendChild(text);
  container.appendChild(item);
}

function roundVal(num, dec = 2) {
  if (num === null || num === undefined) return null;
  return Number(Math.round(num + 'e' + dec) + 'e-' + dec);
}

// 8. Data Export
function exportCsv() {
  if (!state.activeTrafoData) return;
  const data = state.activeTrafoData;
  const t = data.transformer;
  const selectedVers = data.selected_versions;
  const preds = data.predictions;
  const subDates = preds[selectedVers[0]]?.map(p => p.tarih) || [];
  const wxMap = {};
  (data.weather || []).forEach(w => { wxMap[w.tarih] = w.temp_mean ?? w.temp; });

  let csvContent = 'data:text/csv;charset=utf-8,';
  csvContent += `tanim,${t.tanim},guc,${t.guc},lokasyon,"${t.lokasyon}"\n`;
  csvContent += ['tarih', 'day_label', 'temperature_c', 'historical_tuketim', ...selectedVers.map(v => `pred_${v}`)].join(',') + '\n';

  // Historical
  (data.historical || []).forEach(h => {
    const dayLabel = h.day_label || formatDayDow(h.tarih);
    const tempVal = h.temp ?? wxMap[h.tarih] ?? '';
    csvContent += [h.tarih, `"${dayLabel}"`, tempVal, h.tuketim, ...selectedVers.map(() => '')].join(',') + '\n';
  });

  // Forecast
  subDates.forEach(dt => {
    const dayLabel = formatDayDow(dt);
    const tempVal = preds[selectedVers[0]]?.find(x => x.tarih === dt)?.temp ?? wxMap[dt] ?? '';
    const row = [dt, `"${dayLabel}"`, tempVal, ''];
    selectedVers.forEach(v => {
      const p = preds[v]?.find(x => x.tarih === dt)?.tuketim || '';
      row.push(p);
    });
    csvContent += row.join(',') + '\n';
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `trafo_${t.tanim}_comparison.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function exportJson() {
  if (!state.activeTrafoData) return;
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.activeTrafoData, null, 2));
  const link = document.createElement('a');
  link.setAttribute('href', dataStr);
  link.setAttribute('download', `trafo_${state.activeTrafoData.transformer.tanim}_data.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
