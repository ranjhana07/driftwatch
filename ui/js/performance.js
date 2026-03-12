const SERIES_COLORS = { auc: "#2f6fdb", f1: "#3b8f6e", ece: "#e08a1e", rmse: "#d4472b", accuracy: "#8aa4c8" };

export function renderPerformance(chart, noteElement, series) {
  const names = Object.keys(series).filter((name) => name !== "accuracy");
  if (names.length === 0) {
    chart.clear();
    noteElement.textContent = "No labelled windows yet. Labels arrive after the prediction, so these lag behind.";
    return;
  }
  chart.setOption(
    {
      tooltip: { trigger: "axis" },
      legend: { data: names, top: 0, textStyle: { color: "#8d97a8" } },
      grid: { left: 48, right: 16, top: 36, bottom: 40 },
      xAxis: { type: "time" },
      yAxis: { type: "value", scale: true },
      series: names.map((name) => ({
        name,
        type: "line",
        showSymbol: false,
        data: series[name].map(([time, value, flagged]) => ({
          value: [time, value],
          itemStyle: flagged ? { color: "#d4472b" } : undefined,
        })),
        lineStyle: { color: SERIES_COLORS[name] || "#667085", width: 2 },
        itemStyle: { color: SERIES_COLORS[name] || "#667085" },
      })),
    },
    true
  );
  noteElement.textContent = "Computed only from events whose label had arrived by the end of each window.";
}
