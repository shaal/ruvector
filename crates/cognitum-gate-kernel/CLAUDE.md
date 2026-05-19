# cognitum-gate-kernel

No-std WASM kernel for worker tiles in a 256-tile coherence gate fabric. Each tile maintains a local graph shard, accumulates evidence via sequential testing (e-values), and produces witness fragments that aggregate to a global min-cut. Targets a ~64 KB memory budget per tile so 256 tiles fit a typical edge device.

## Layout

- `Cargo.toml` — lib + cdylib; `default = ["std"]`; `std` brings in path dep `ruvector-mincut` (with `wasm` feature). `canonical-witness` feature toggles pseudo-deterministic witness fragments.
- `SECURITY.md` — threat model and audit notes (large file, ~55 KB).
- `src/lib.rs` — crate root, declares `no_std` when `std` feature off, sets up global allocator for bare WASM, re-exports the tile API (`TileState`, `ingest_delta`, `tick`, `get_witness_fragment`).
- `src/shard.rs` — `CompactGraph` (~42 KB budget): vertices, edges, adjacency for the local shard.
- `src/delta.rs` — `Delta` variants (edge add/remove/weight change) consumed by `ingest_delta`.
- `src/evidence.rs` — `EvidenceAccumulator`: e-value hypothesis tests with sliding window (~2 KB).
- `src/report.rs` — tick reports rolled up from graph + evidence state.
- `src/canonical_witness.rs` — canonical (pseudo-deterministic) witness fragment emission (gated by `canonical-witness`).
- `benches/benchmarks.rs` — criterion harness (declared in `[[bench]]`).
- `tests/` — `security_tests.rs`, `canonical_witness_bench.rs`.
- `tests_disabled/` — quarantined tests (`evidence_tests`, `integration`, `report_tests`, `shard_tests`); kept for future re-enablement.
- `docs/SECURITY_AUDIT.md` — audit report.

## Public API

`TileState::new(id) -> TileState`, `ingest_delta(&Delta)`, `tick(epoch) -> Report`, `get_witness_fragment() -> WitnessFragment`. Types `Delta`, `Report`, `WitnessFragment`, `EvidenceAccumulator`, `CompactGraph` are re-exported from `lib.rs`.

## Related

- `crates/ruvector-mincut` — host-side aggregator; shared type definitions via the `wasm` feature.
- `crates/ruvector-dag/src/mincut` — DAG-level mincut engine that consumes aggregated witnesses.
- `crates/ruvector-mincut-gated-transformer-wasm` — sibling WASM crate that gates inference on the same signals.
