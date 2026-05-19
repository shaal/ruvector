# ruvector-metrics

Prometheus-compatible metrics collection for ruvector vector databases. Provides counters, histograms and gauges for search/insert/delete throughput and latency, plus a health-check abstraction.

## Layout

- `Cargo.toml` — depends on `prometheus`, `lazy_static`, `serde`, `serde_json`, `chrono`.
- `src/lib.rs` — declares the global `REGISTRY` plus `SEARCH_REQUESTS_TOTAL`, `SEARCH_LATENCY_SECONDS`, `INSERT_REQUESTS_TOTAL`, `INSERT_LATENCY_SECONDS`, and friends as `lazy_static!` Prometheus metrics. Re-exports `MetricsRecorder` and the `HealthChecker` API.
- `src/recorder.rs` — `MetricsRecorder` helper for recording timed search/insert/delete events.
- `src/health.rs` — `HealthChecker`, `HealthStatus`, `HealthResponse`, `ReadinessResponse`, `CollectionHealth` for `/healthz` and `/readyz` endpoints.

## Public API

- `REGISTRY` and the global `*_TOTAL` / `*_SECONDS` metric handles
- `MetricsRecorder`
- Health-check types: `HealthChecker`, `HealthStatus`, `HealthResponse`, `ReadinessResponse`, `CollectionHealth`
- `TextEncoder` re-exported for scrape-endpoint use

## Related

- `../ruvector-server` — HTTP server that exposes a `/metrics` route using these primitives
- `../ruvector-core` — the vector DB whose ops are recorded
