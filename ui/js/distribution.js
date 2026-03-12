import { formatTime } from "./models.js";

function binLabels(edges) {
  const labels = [];
  for (let i = 0; i < edges.length - 1; i += 1) {
    labels.push(((edges[i] + edges[i + 1]) / 2).toPrecision(3));
  }
  return labels;
}

export function renderDistribution(chart, noteElement, data) {
  const numeric = data.kind === "numeric";
  const labels = numeric ? binLabels(data.edges) : data.categories;
  chart.setOption(
    {
      tooltip: { trigger: "axis", valueFormatter: (value) => `${(value * 100).toFixed(1)}%` },
      legend: { data: ["reference", "current"], top: 0, textStyle: { color: "#8d97a8" } },
      grid: { left: 52, right: 16, top: 36, bottom: 40 },
      xAxis: { type: "category", data: labels, axisLabel: { hideOverlap: true, rotate: numeric ? 0 : 25 } },
      yAxis: { type: "value", axisLabel: { formatter: (value) => `${Math.round(value * 100)}%` } },
      series: [
        { name: "reference", type: "bar", data: data.reference, itemStyle: { color: "#8aa4c8" }, barGap: "-100%", z: 1 },
        {
          name: "current",
          type: numeric ? "line" : "bar",
          data: data.current,
          smooth: numeric,
          showSymbol: false,
          itemStyle: { color: "#d4472b" },
          lineStyle: { width: 2 },
          z: 2,
        },
      ],
    },
    true
  );
  noteElement.textContent =
    `${data.feature}: reference n=${data.n_reference.toLocaleString()}, ` +
    `current n=${data.n_current.toLocaleString()} (window ending ${formatTime(data.window_end)})`;
}

export function clearDistribution(chart, noteElement, message) {
  chart.clear();
  noteElement.textContent = message;
}
