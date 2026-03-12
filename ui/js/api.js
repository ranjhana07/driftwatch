const params = new URLSearchParams(window.location.search);
const base = params.get("api") || "";

let apiKey = "";
try {
  apiKey = window.localStorage.getItem("driftwatch.apiKey") || "";
} catch (err) {
  apiKey = "";
}

export function setApiKey(value) {
  apiKey = value;
  try {
    window.localStorage.setItem("driftwatch.apiKey", value);
  } catch (err) {
    return;
  }
}

export function getApiKey() {
  return apiKey;
}

async function request(path, query = {}) {
  const url = new URL(base + path, window.location.origin);
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  const headers = apiKey ? { "X-API-Key": apiKey } : {};
  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${path}`);
  }
  return response.json();
}

export const api = {
  models: () => request("/v1/models"),
  heatmap: (model) => request(`/v1/models/${encodeURIComponent(model)}/heatmap`),
  distribution: (model, feature, windowEnd) =>
    request(`/v1/models/${encodeURIComponent(model)}/distribution`, { feature, window_end: windowEnd }),
  performance: (model) => request(`/v1/models/${encodeURIComponent(model)}/performance`),
  alerts: (model) => request("/v1/alerts", { model, limit: 200 }),
};
