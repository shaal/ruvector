# wasm/ios/benches

Benchmark and simulation binaries for `ruvector-ios-wasm`. These are wired as `[[bin]]` entries in `../Cargo.toml`, not Criterion harnesses.

## Files

- `performance.rs` - Native performance benchmark across HNSW/quantization/distance.
- `ios_simulation.rs` (~38 KB) - Simulates iOS workloads (memory caps, thermal, battery) on the host.

## How to run

```bash
cargo run --bin benchmark --release
cargo run --bin ios_simulation --release
```

## Related

- Tests: `../tests/engine_tests.rs`.
- Build profiles: `../Cargo.toml`.
