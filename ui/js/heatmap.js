import { formatTime } from "./models.js";

const OUTPUT_LABELS = { __embedding: "embedding", __score: "model score", __prediction: "predicted class" };

function featureLabel(name) {
  return OUTPUT_LABELS[name] || name;
}

const LEVEL_COLORS = ["#d9e4df", "#e8d27a", "#e9a24a", "#d4472b", "#8b1a1a"];

export function renderHeatmap(chart, data, onCellClick) {
  if (data.windows.length === 0) {
    chart.clear();
    chart.setOption({
      title: { text: "No monitor results yet", left: "center", top: "middle", textStyle: { color: "#8d97a8", fontSize: 14 } },
    });
    return;
  }
  const cells = data.cells.map(([x, y, severity, flagged, metric]) => ({
    value: [x, y, severity],
    flagged,
    metric,
  }));
  chart.setOption(
    {
      title: { show: false },
      tooltip: {
        position: "top",
        formatter: (point) => {
          const [x, y, severity] = point.value;
          const state = point.data.flagged ? "flagged" : "ok";
          return `${featureLabel(data.features[y])}<br>${formatTime(data.windows[x])}<br>severity ${severity.toFixed(2)} (${point.data.metric}, ${state})`;
        },
      },
      grid: { left: 130, right: 24, top: 12, bottom: 78 },
      xAxis: {
        type: "category",
        data: data.windows.map(formatTime),
        splitArea: { show: false },
        axisLabel: { hideOverlap: true, rotate: 30 },
      },
      yAxis: { type: "category", data: data.features, inverse: true, axisLabel: { interval: 0, formatter: featureLabel } },
      visualMap: {
        type: "piecewise",
        orient: "horizontal",
        left: "center",
        bottom: 0,
        pieces: [
          { min: 0, max: 0.25, label: "none", color: LEVEL_COLORS[0] },
          { min: 0.25, max: 0.5, label: "low", color: LEVEL_COLORS[1] },
          { min: 0.5, max: 0.75, label: "medium", color: LEVEL_COLORS[2] },
          { min: 0.75, max: 0.9, label: "high", color: LEVEL_COLORS[3] },
          { min: 0.9, max: 1.01, label: "critical", color: LEVEL_COLORS[4] },
        ],
        textStyle: { color: "#8d97a8" },
      },
      dataZoom: [{ type: "inside", xAxisIndex: 0 }],
      series: [
        {
          type: "heatmap",
          data: cells,
          itemStyle: { borderColor: "rgba(120,130,150,0.25)", borderWidth: 1 },
          emphasis: { itemStyle: { borderColor: "#2f6fdb", borderWidth: 2 } },
        },
      ],
    },
    true
  );
  chart.off("click");
  chart.on("click", (point) => {
    const [x, y] = point.value;
    onCellClick(data.features[y], data.windows[x]);
  });
}
