# ruvector-metrics/src

Source for the Prometheus metrics layer.

## Files

- `lib.rs` — global `REGISTRY` and `lazy_static!` metric handles (search/insert latency histograms and request counters labelled by collection+status). Module decls + re-exports.
- `recorder.rs` — `MetricsRecorder` — convenience facade for recording timed events into the global metrics.
- `health.rs` — `HealthChecker`, status enums, and JSON response structs for liveness/readiness endpoints.
