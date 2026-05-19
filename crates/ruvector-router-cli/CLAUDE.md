# ruvector-router-cli

Single-binary CLI for testing and benchmarking `../ruvector-router-core`. Binary name is `ruvector`.

## Layout

- `Cargo.toml` — `[[bin]] name = "ruvector" path = "src/main.rs"`. Deps: `ruvector-router-core`, `clap` (derive), `colored`, `rand`, `chrono`, `serde_json`, `anyhow`, `tracing`/`tracing-subscriber`. Release profile: LTO, opt 3, single codegen unit.
- `src/main.rs` — clap `Cli` with subcommands invoking `ruvector_router_core::{VectorDB, VectorEntry, SearchQuery, DistanceMetric}` (create collection, insert, search, benchmark, etc.).

## Public API

Binary only — no library surface.

## Related

- `../ruvector-router-core` — the routed vector DB this CLI exercises
- `../ruvector-bench` — heavier criterion-style benchmarks
