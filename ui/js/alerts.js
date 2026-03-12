import { formatTime } from "./models.js";

const LEVEL_COLORS = { low: "#c9a227", medium: "#e08a1e", high: "#d4472b", critical: "#8b1a1a" };
const LEVEL_ROWS = ["low", "medium", "high", "critical"];

export function renderAlerts(chart, table, alerts) {
  const body = table.querySelector("tbody");
  body.replaceChildren();
  if (alerts.length === 0) {
    chart.clear();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "empty";
    cell.textContent = "No alerts raised.";
    row.append(cell);
    body.append(row);
    return;
  }
  chart.setOption(
    {
      tooltip: {
        formatter: (point) => `${formatTime(point.value[0])}<br>${point.data.rule}<br>${point.data.message}`,
      },
      grid: { left: 70, right: 16, top: 10, bottom: 28 },
      xAxis: { type: "time" },
      yAxis: { type: "category", data: LEVEL_ROWS },
      series: [
        {
          type: "scatter",
          symbolSize: 12,
          data: alerts.map((alert) => ({
            value: [alert.created_at, alert.level],
            rule: alert.rule,
            message: alert.message,
            itemStyle: { color: LEVEL_COLORS[alert.level] || "#667085" },
          })),
        },
      ],
    },
    true
  );
  alerts.slice(0, 50).forEach((alert) => {
    const row = document.createElement("tr");
    const time = document.createElement("td");
    time.textContent = formatTime(alert.created_at);
    const level = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `badge ${alert.level}`;
    badge.textContent = alert.level;
    level.append(badge);
    const rule = document.createElement("td");
    rule.textContent = alert.rule;
    const message = document.createElement("td");
    message.textContent = alert.message;
    row.append(time, level, rule, message);
    body.append(row);
  });
}
