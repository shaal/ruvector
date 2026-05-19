# wasm/ios

`ruvector-ios-wasm` (standalone crate, own workspace): iOS- and browser-optimized WASM vector database with HNSW, quantization, distance metrics, attention, embeddings, Q-learning, and iOS-specific capabilities + on-device learning. Ships Swift bindings, TypeScript types, and a build script.

## Files

- `Cargo.toml` - Standalone package; features `browser`, `simd`, `full`. Tight size profile (`opt-level="z"`, `lto="fat"`, `strip="symbols"`, `panic=abort`). Defines benchmark bins `benchmark` and `ios_simulation`.
- `Cargo.lock` - Lockfile.
- `src/` - Rust library (HNSW, distance, embeddings, attention, quantization, qlearning, simd, ios_capabilities, ios_learning).
- `benches/performance.rs`, `benches/ios_simulation.rs` - Benchmark / simulation binaries.
- `tests/engine_tests.rs` - Engine integration tests.
- `dist/recommendation.wasm` - Pre-built WASM artifact.
- `scripts/build.sh` - Multi-target build script.
- `swift/` - Swift package + iOS service code consuming the WASM.
- `types/` - npm package with TypeScript declarations.

## How to build/run

```bash
cd /home/user/ruvector/examples/wasm/ios
bash scripts/build.sh
# Native benches:
cargo run --bin benchmark --release
cargo run --bin ios_simulation --release
cargo test
```

## Tech stack

- Rust 2021 (no_std-friendly defaults), optional `wasm-bindgen`/`js-sys`/`web-sys`, serde/serde-wasm-bindgen, WASM SIMD.
- Swift Package consumer + TypeScript types for JS consumers.

## Related

- Other WASM crates: `examples/onnx-embeddings-wasm`, `examples/prime-radiant/wasm`, `examples/scipix/src/wasm`.
- HNSW reference: `examples/data/framework/src/hnsw.rs`.
