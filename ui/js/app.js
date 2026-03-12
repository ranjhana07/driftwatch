import { api, getApiKey, setApiKey } from "./api.js";
import { renderAlerts } from "./alerts.js";
import { clearDistribution, renderDistribution } from "./distribution.js";
import { renderHeatmap } from "./heatmap.js";
import { formatTime, renderModels } from "./models.js";
import { renderPerformance } from "./performance.js";

const state = { model: null, feature: null, windowEnd: null, timer: null };

const els = {
  models: document.getElementById("models"),
  detail: document.getElementById("detail"),
  title: document.getElementById("model-title"),
  meta: document.getElementById("model-meta"),
  featureSelect: document.getElementById("feature-select"),
  distributionNote: document.getElementById("distribution-note"),
  performanceNote: document.getElementById("performance-note"),
  alertsTable: document.getElementById("alerts-table"),
  apiKey: document.getElementById("api-key"),
  autoRefresh: document.getElementById("auto-refresh"),
  refresh: document.getElementById("refresh"),
};

const charts = {
  heatmap: echarts.init(document.getElementById("heatmap")),
  distribution: echarts.init(document.getElementById("distribution")),
  performance: echarts.init(document.getElementById("performance")),
  alerts: echarts.init(document.getElementById("alerts-timeline")),
};

window.addEventListener("resize", () => Object.values(charts).forEach((chart) => chart.resize()));

function showError(error) {
  els.models.replaceChildren();
  const message = document.createElement("div");
  message.className = "empty";
  message.textContent = `Could not reach the collector: ${error.message}`;
  els.models.append(message);
}

async function loadDistribution() {
  if (!state.feature) {
    clearDistribution(charts.distribution, els.distributionNote, "No feature selected.");
    return;
  }
  try {
    const data = await api.distribution(state.model, state.feature, state.windowEnd);
    renderDistribution(charts.distribution, els.distributionNote, data);
  } catch (error) {
    clearDistribution(charts.distribution, els.distributionNote, error.message);
  }
}

function selectFeature(feature, windowEnd) {
  state.feature = feature;
  state.windowEnd = windowEnd || null;
  els.featureSelect.value = feature;
  loadDistribution();
}

function fillFeatureSelect(features) {
  els.featureSelect.replaceChildren();
  features.forEach((feature) => {
    const option = document.createElement("option");
    option.value = feature;
    option.textContent = feature;
    els.featureSelect.append(option);
  });
  if (!features.includes(state.feature)) {
    state.feature = features[0] || null;
  }
  if (state.feature) {
    els.featureSelect.value = state.feature;
  }
}

async function loadDetail() {
  if (!state.model) {
    els.detail.hidden = true;
    return;
  }
  els.detail.hidden = false;
  els.title.textContent = state.model;
  const [heat, performance, alerts] = await Promise.all([
    api.heatmap(state.model),
    api.performance(state.model),
    api.alerts(state.model),
  ]);
  els.meta.textContent = heat.windows.length
    ? `${heat.windows.length} windows, latest ${formatTime(heat.windows[heat.windows.length - 1])}`
    : "no windows evaluated";
  fillFeatureSelect(heat.features);
  renderHeatmap(charts.heatmap, heat, selectFeature);
  renderPerformance(charts.performance, els.performanceNote, performance);
  renderAlerts(charts.alerts, els.alertsTable, alerts);
  Object.values(charts).forEach((chart) => chart.resize());
  await loadDistribution();
}

async function loadModels() {
  try {
    const models = await api.models();
    if (!state.model && models.length > 0) {
      state.model = models[0].name;
    }
    renderModels(els.models, models, state.model, (name) => {
      state.model = name;
      state.feature = null;
      state.windowEnd = null;
      refresh();
    });
    await loadDetail();
  } catch (error) {
    showError(error);
  }
}

function refresh() {
  return loadModels();
}

els.featureSelect.addEventListener("change", () => selectFeature(els.featureSelect.value, null));
els.refresh.addEventListener("click", refresh);
els.apiKey.value = getApiKey();
els.apiKey.addEventListener("change", () => {
  setApiKey(els.apiKey.value.trim());
  refresh();
});
els.autoRefresh.addEventListener("change", () => {
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
  if (els.autoRefresh.checked) {
    state.timer = window.setInterval(refresh, 30000);
  }
});

refresh();
