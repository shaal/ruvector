# delta-behavior

Reference implementation of *delta-behavior*: constrained state transitions that preserve global coherence - "systems that refuse to collapse". Provides core Rust primitives, 10 applied demos, benches, a WASM SDK, and a long-form whitepaper / ADR set.

## Important files
- `Cargo.toml` - `cdylib + rlib` crate. Feature-gated applications (`self-limiting-reasoning`, `event-horizon`, `homeostasis`, `world-model`, `coherence-creativity`, `anti-cascade`, `graceful-aging`, `swarm-intelligence`, `graceful-shutdown`, `containment`) plus rollup groups (`all-applications`, `safety-critical`, `distributed`, `ai-ml`). Targets both native and `wasm32`.
- `Cargo.lock` - committed lockfile.
- `WHITEPAPER.md`, `CHANGELOG.md`, `LICENSE-MIT`, `LICENSE-APACHE` - distribution metadata.
- `src/` - library code (`lib.rs`, `simd_utils.rs`, `wasm.rs`, `bin/run_benchmarks.rs`).
- `applications/` - the 10 numbered application demos (`01-self-limiting-reasoning.rs` ... `11-extropic-substrate.rs`).
- `examples/demo.rs` - minimal getting-started demo (`cargo run --example demo`).
- `benches/coherence_benchmarks.rs` - Criterion benches (feature `benchmarks`).
- `tests/` - integration tests (native + WASM bindings).
- `wasm/` - TypeScript SDK + `wasm-pack` packaging on top of the Rust crate.
- `adr/`, `ddd/`, `docs/`, `research/`, `scripts/` - architecture, domain model, API/security docs, theory, and `build-wasm.sh`.

## Build / run
- Library: `cargo build -p delta-behavior --features full`.
- Demo: `cargo run -p delta-behavior --example demo`.
- Application demo (e.g.): `cargo run -p delta-behavior --example swarm --features swarm-intelligence`.
- Benches: `cargo bench -p delta-behavior --features benchmarks`.
- WASM SDK: `cd wasm && npm run build:all`.

## Related
- Used by `../OSpipe/` (workspace path dep `ruvector-delta-core`).
- Companion research-tier demos: `../cmb-consciousness/`, `../brain-boundary-discovery/`.
