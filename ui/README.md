# ui

Static dashboard for the collector API. No build step: plain HTML, CSS and ES modules, with ECharts loaded from a CDN.

| File | Purpose |
| --- | --- |
| `index.html` | page layout |
| `css/style.css` | styles, light and dark |
| `js/api.js` | fetch wrapper, optional `X-API-Key` header |
| `js/models.js` | model cards with the latest status |
| `js/heatmap.js` | feature drift heatmap over time |
| `js/distribution.js` | reference vs current distribution for one feature |
| `js/performance.js` | AUC, F1, calibration error or RMSE over labelled windows |
| `js/alerts.js` | alert timeline and table |
| `js/app.js` | state and wiring |

## Running it

Same origin as the collector:

```
driftwatch serve --ui-dir ui
```

Then open http://127.0.0.1:8000. The compose file serves it from nginx and proxies `/v1` to the collector instead.

To point a static copy at another collector, add `?api=http://host:8000` to the URL. The collector has to allow your origin if it is not the same host. Clicking a heatmap cell loads that feature and window in the distribution panel.
