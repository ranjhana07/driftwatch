# deploy

Configuration mounted by `docker-compose.yml`.

| Path | Used by | Notes |
| --- | --- | --- |
| `alert_rules.yaml` | scheduler | copied into the image, path set by `DRIFTWATCH_ALERT_RULES_PATH` |
| `prometheus/prometheus.yml` | prometheus | scrapes the collector on `:8000/metrics` and the scheduler on `:9102` |
| `nginx/default.conf` | ui | serves `ui/` and proxies `/v1/` to the collector |
| `grafana/provisioning/` | grafana | Prometheus datasource and a file provider for dashboards |
| `grafana/dashboards/driftwatch.json` | grafana | one overview dashboard with a `model` variable |

## Alert rules

```yaml
rules:
  - name: drift-high
    model: "*"
    kinds: [drift]
    metrics: [psi, ks]
    features: [age]
    min_level: high
    consecutive: 2
    cooldown_seconds: 3600
    notifiers: [slack]
```

`min_level` is one of `low`, `medium`, `high`, `critical`. `consecutive` requires the rule to match in that many consecutive windows before an alert is raised. `cooldown_seconds` is the minimum gap between two alerts from the same rule on the same model. Without `notifiers` the alert goes to every configured notifier.

Slack needs `DRIFTWATCH_SLACK_WEBHOOK_URL`. Email needs `DRIFTWATCH_SMTP_HOST` and `DRIFTWATCH_SMTP_TO` (comma separated).

## Grafana

The dashboard is provisioned from the mounted folder and can be edited in the UI. To change the committed version, export the JSON and replace `grafana/dashboards/driftwatch.json`.
