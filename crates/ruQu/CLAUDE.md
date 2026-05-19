# ruQu

Classical nervous system for quantum machines — real-time syndrome processing and coherence assessment via dynamic min-cut. Provides high-throughput, low-latency pipelines that ingest quantum error syndromes and transform them into coherence-relevant signals using a two-layer classical control approach (RuVector memory layer + dynamic min-cut gate).

## Important files

- `Cargo.toml` — Package `ruqu` (note lowercase). Optional deps: `ruvector-mincut` (exact mode), `cognitum-gate-tilezero`, `fusion-blossom` (QEC decoding), `ruvector-mincut-gated-transformer`, `rayon`, `tracing`. Cryptography via `blake3`, `ed25519-dalek`, `subtle`. Graph via `petgraph`.
- `src/lib.rs` — Crate root. Doc-comment describes DDD bounded contexts: Syndrome Processing, Coherence Gate, Tile Architecture (256-tile WASM fabric).
- `docs/RESEARCH_DISCOVERIES.md`, `docs/SECURITY-REVIEW.md`, `docs/SIMULATION-INTEGRATION.md` — Research notes.
- `docs/adr/ADR-001-ruqu-architecture.md` — Top-level architecture decision.
- `docs/ddd/DDD-001-coherence-gate-domain.md`, `DDD-002-syndrome-processing-domain.md` — DDD context maps.

## Source modules (`src/`)

- `syndrome.rs`-equivalents split across files:
  - `adaptive.rs` — Adaptive thresholding.
  - `attention.rs` — Attention-based gating.
  - `decoder.rs` — Syndrome decoder.
  - `fabric.rs` — 256-tile WASM fabric.
  - `filters.rs` — Signal filters.
  - `metrics.rs` — Telemetry/metrics.
  - `mincut.rs` — Dynamic min-cut coherence gate.
  - `parallel.rs` — Rayon-based parallelism.
  - `schema.rs` — Wire/data schema.
  - `stim.rs` — Integration with Stim quantum simulator.
  - `syndrome.rs` — `DetectorBitmap`, `SyndromeRound`, `SyndromeBuffer` (per lib.rs doctest).
  - `tile.rs` — Tile primitives.
  - `traits.rs` / `types.rs` — Public trait/types.
  - `error.rs` — `Error` enum.
- `src/bin/` — `ruqu_demo.rs`, `ruqu_predictive_eval.rs` (CLI demos).

## Tests / Benches / Examples

- `tests/` — `filter_tests.rs`, `integration_tests.rs`, `stress_tests.rs`, `syndrome_tests.rs`, `tile_tests.rs`.
- `benches/` — `latency_bench.rs`, `memory_bench.rs`, `mincut_bench.rs`, `scaling_bench.rs`, `syndrome_bench.rs`, `throughput_bench.rs`.
- `examples/` — Coherence-gate simulations, MWPM comparison benchmark, quantum fabric basics, QEC integration.

## Related

- `ruvector-mincut` (algorithmic backend), `ruvector-mincut-gated-transformer`, `cognitum-gate-tilezero`.
- Sister experimental crate: `ruqu-exotic` (exotic quantum/classical hybrids).
- WASM browser companion: `ruqu-wasm`.
