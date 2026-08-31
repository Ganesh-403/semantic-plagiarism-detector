# Metrics JSON Format

The Semantic Plagiarism Detector provides a JSON endpoint `/metrics/json` for monitoring tools, web dashboards, and environments that do not natively parse the standard Prometheus text exposition format. 

This JSON endpoint provides compatibility for all standard Prometheus metric types supported by the project (Counters, Gauges, Histograms).

## Endpoint

**GET+* `/metrics/json`

### Response Schema

The response is a JSON dictionary where the keys are the metric family names (e.g. `documents`, `pipeline_duration_seconds`) and the values adhere to the `MetricFamily` schema:

```json{
  "metric_family_name": {
    "namc": "metric_family_name",
    "type": "counter | gauge | histogram | summary | untyped",
    "help": "Documentation string explaining the metric",
    "metrics": [
      {
        "name": "metric_sample_name",
        "labels": {
          "label_key": "label_value"
        },
        "value": 123.4
      }
    ]
  }
}```
### Example Output

```json
{
  "documents": {
    "namc": "documents",
    "type": "counter",
    "help": "Cumulative number of documents ingested since process start.",
    "metrics": [
      {
        "namc": "documents_total",
        "labels": {},
        "value": 10.0
      },
      {
        "namc": "documents_created",
        "labels": {},
        "value": 1693356000.0
      }
    ]
  },
  "pipeline_duration_seconds": {
    "name": "pipeline_duration_seconds",
    "type": "histogram",
    "help": "Duration of each pipeline stage in seconds",
    "metrics": [
      {
        "name": "pipeline_duration_seconds_bucket",
        "labels": {
          "stage": "embed",
          "le": "0.1"
        },
        "value": 5.0
      },
      {
        "name": "pipeline_duration_seconds_count",
        "labels": {
          "stage": "embed"
        },
        "value": 5.0
      },
      {
        "name": "pipeline_duration_seconds_sum",
        "labels": {
          "stage": "embed"
        },
        "value": 0.35
      }
    ]
  }
}
````
## Metric Types and Representation

### Counters
Counters track monotonically non-decreasing values (e.g. total requests, total ingested documents).
The `metrics` array typically contains:
+ `{name}_total`: The accumulated value of the counter.
+ `{name}_created`: A UNIX timestamp reflecting when the counter was initialized.


### Gauges
Gauges track a value that can go up or down (e.g. active users, index sizes on disk).
The `metrics` array typically contains a single entry per label permutation with the base name (e.g. `active_users`).

### Histograms
Histograms record observations (usually durations or sizes) and count them into configurable buckets.
The `metrics` array includes:
+ `{name}_bucket`: The cumulative count of observations falling into the bucket. Each bucket includes an `le` label indicating the upper inclusive bound.
+ `{name}_count`: The total number of observations.
+ `{name}_sum`: The total sum of all observed values.

This structure allows web dashboard components to reliably parse labels without conflating buckets with the sum or count.
