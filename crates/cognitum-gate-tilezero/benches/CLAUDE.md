# cognitum-gate-tilezero/benches

Criterion benchmarks for the TileZero arbiter. Run with `cargo bench -p
cognitum-gate-tilezero`. Uses `criterion` with `html_reports` and `async_tokio`.

## Files
- `decision_bench.rs` - declared in Cargo.toml as the primary `[[bench]]`;
  measures throughput of the three-filter decision pipeline.
- `merge_bench.rs` - worker report merging throughput.
- `crypto_bench.rs` - Ed25519 sign/verify and Blake3 hashing micro-benchmarks
  on the permit and receipt paths.
- `benchmarks.rs` - end-to-end gate latency across realistic worker counts.
