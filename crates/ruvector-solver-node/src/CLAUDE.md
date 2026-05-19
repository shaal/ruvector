# ruvector-solver-node/src

- `lib.rs` — single source file. Defines the NAPI-facing types (`SolveConfig`, `SolveResult`, plus PageRank / complexity-estimation structs) and the `#[napi]` async wrappers that route into `ruvector_solver` on a `tokio::task::spawn_blocking` worker.

See `../CLAUDE.md`.
