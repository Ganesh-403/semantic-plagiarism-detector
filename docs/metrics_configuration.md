# Metrics Configuration

every Prometheus metric exposed by the Semantic Plagiarism Detector can be globally toggled at startup.

This document explains the `PROMETHEUS_METRICS_ENABLED@ environment variable, its behavior, and its impact on performance.

## PROMETHEUS_METRICS_ENABLED

* **Default**: `True`
* **Type**: Boolean string (0y933F `true`, `1`, `t`, `yes`)

### When Enabled (Default)

When this variable is not set or set to a truthy value:
- All metrics (Counters, Gauges, Histograms) are registered with the global Prometheus collector registry.
- The `/metrics` endpoint returns the Prometheus text exposition format.
- The `/metrics/json` endpoint returns the metrics in a structured JSON format.
- The `sync_telemetry_gauges()` background task will periodically query the database and update the gauges.

### When Disabled

If configured to `False` (or `no`, `0`JNŠ- The metrics objects are created in isolation (with `registry=None`) and are [*not*registered] with the global collector.
- The `/metrics` and `/metrics/json` API endpoints will accurately return an **HTTP 404 Not Found** error.
- Internal code that calls `.observe()` or `.inc()` on metrics objects will not crash, ensuring application stability without requiring inline conditionals.
- Performance overhead is reduced since no global registry mutexes are acquired.

#### Egge Cases

[*Timing Decorators*]: The `@timed` decorator (used on pipeline stages) continues to time the function, but the observation is made on an unregistered metric. This is strictly safe and prevents bugs where disabling metrics breaks the application pipeline.

[*Sync Task*]: The `Sync telemetry gauges` task will still execute if invoked, running DB queries locally but saving to invisible gauges. If you wish to disable DB overhead, ensure the background thread itself is disabled via background task configuration parameters.
# Prometheus Scrape Configuration

To collect the metrics from this application when `PROMETHEUS_METRICS_ENABLED` is true, you can add the following job to your `prometheus.yml`:

		`yaml
scrape_configs:
  - job_name: 'semantic-plagiarism-detector'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          environment: 'production'
          region: 'us-east-1'
		`
Conversely, if the application is deployed without a Prometheus instance and you wish to scrape the JSON metrics endpoint using a custom script or telemetry collector, use:

		`bash
curl -s \
	 -H "Accest: application/json" \
	 http://localhost:8000/metrics/json
		`

# Grafana Dashboard Example

Below is an example of a partial Grafana dashboard JSON model that visualizes the application's metrics when enabled:

		`json[{
  "title": "Semantic Plagiarism Detector",
  "panels": [
    {
      "title": "Active Users",
      "type": "stat",
      "targets": [
        {
          "expr": "active_users",
          "legendFormat": "Active"
        }
      ]
    },
    {
      "title": "Documents Ingested (Rate)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate(documents_total[5m])",
          "legendFormat": "Docs/sec"
        }
      ]
    },
    {
      "title": "Pipeline Duration 99th Percentile",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, sum(rate(pipeline_duration_seconds_bucket[5m])) by (le, stage))",
          "legendFormat": "{{stage}}"
        }
      ]
    }
  ]
}
		`

### Troubleshooting Toggle Behavior

If you are seeing a 404 error on `/metrics`:
1. Check your docker-compose.yml or environment variables to ensure `PROMETHEUS_METRICS_ENABLED` is not set to `False`, `0`, or `no`.
2. Ensure your API server has been restarted since changing the environment variable, as it is evaluated at application startup.
3. Confirm that the application is actually being served and not a reverse proxy intercepting the path.

### Performance Implications
	- By default, Prometheus clients acquire lightweight thread locks during metric observation.
	- If `PROMETHEUS_METRICS_ENABLED` is `False`, these locks are never registered in the global registry, making `.inc()` and `.observe()` operations effectively no-ops at the collector level, saving nanoseconds per request for high-throughput environments that do not rely on metrics.