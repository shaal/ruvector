# neural-trader-coherence

Implements the MinCut coherence gate, CUSUM drift detection, and proof-gated mutation protocol for the RuVector Neural Trader (ADR-084). Every memory write, model update, retrieval, and actuation in the trader passes through this gate before it is allowed to proceed.

Internal-only (`publish = false`).

## Layout

- `Cargo.toml` — tiny, depends only on `anyhow` and `serde`.
- `src/lib.rs` — entire crate in a single file: gate types and the evaluation loop.

## Public API / key types

- `CoherenceDecision { allow_retrieve, allow_write, allow_learn, allow_act, mincut_value, partition_hash, drift_score, cusum_score, reasons }` with `all_allowed()` / `fully_blocked()` helpers.
- `GateContext { symbol_id, venue_id, ts_ns, mincut_value, partition_hash, cusum_score, drift_score, regime, ... }` — input to the gate.
- `RegimeLabel` enum and the per-action gate evaluation function.

All types derive `Serialize`/`Deserialize` so decisions can be witnessed.

## Related

- `crates/ruvector-dag/src/mincut` — DAG-level mincut engine providing the `mincut_value` signal.
- `crates/cognitum-gate-kernel` — WASM tile-level gate that feeds aggregated witnesses into this trader gate.
- `crates/ruvector-math/src/spectral` and `optimal_transport` — drift/CUSUM signal sources.
