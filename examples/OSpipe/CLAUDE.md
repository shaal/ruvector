# OSpipe

OSpipe is a RuVector-enhanced personal AI memory system designed for Screenpipe integration. It captures local OS activity (frames, audio, etc.), runs a hybrid vector + graph search pipeline, and exposes both a native HTTP server (`ospipe-server`) and a WebAssembly build for in-browser use.

## Important files
- `Cargo.toml` - dual-target crate (native + `wasm32-unknown-unknown`); native pulls in `ruvector-core/filter/cluster/delta-core/router-core/graph/gnn/attention`, `cognitum-gate-kernel`, `ruqu-algorithms` plus `axum`/`tokio` for the HTTP server. WASM target uses `wasm-bindgen`/`js-sys`.
- `ADR-OSpipe-screenpipe-integration.md` - architecture decision record (~88 KB) covering the screenpipe integration design.
- `.github-ci-stub.yml` - reference CI workflow.
- `src/lib.rs`, `src/config.rs`, `src/error.rs`, `src/persistence.rs`, `src/safety.rs` - library root and shared utilities.
- `src/bin/`, `src/capture/`, `src/graph/`, `src/learning/`, `src/pipeline/`, `src/quantum/`, `src/search/`, `src/server/`, `src/storage/`, `src/wasm/` - subsystems (see each subdir's CLAUDE.md).
- `dist/` - prebuilt binaries (linux x86_64/arm64, windows) plus npm tarballs.
- `tests/integration.rs`, `tests/wasm.rs` - integration tests for native and wasm targets.

## Build / run
- `cargo build --release -p ospipe` then `./target/release/ospipe-server` for the HTTP server.
- WASM: `wasm-pack build --target web` (cdylib crate-type already set).

## Related siblings
- `../delta-behavior/` for the coherence primitives, `../edge/` for swarm-style P2P + WASM patterns, `../ruvLLM/` for the LLM pieces this can plug into.
