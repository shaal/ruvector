# cognitum-gate-tilezero/src

TileZero arbiter implementation: assembles worker reports, runs the three-filter
decision pipeline, mints signed permits, and writes receipts.

## Files
- `lib.rs` - top-level re-exports and shared scalar types (`ActionId`, `VertexId`,
  `EdgeId`, `TileId`); atomic counters used across modules.
- `decision.rs` - gate decision logic: `GateDecision`, `GateThresholds`, the
  three-filter chain (`ThreeFilterDecision`), and outcome enum.
- `evidence.rs` - evidence filter (filter 2 of the pipeline); aggregates per-tile
  evidence votes into `AggregatedEvidence`.
- `merge.rs` - merges `WorkerReport`s into a single `MergedReport` using a
  configurable `MergeStrategy`.
- `permit.rs` - Ed25519 token mint/verify; `PermitToken`, `Verifier`,
  `TokenDecodeError`, `VerifyError`.
- `receipt.rs` - hash-chained receipt log (`ReceiptLog`, `WitnessReceipt`,
  `TimestampProof`, `WitnessSummary`).
- `replay.rs` - replay/audit utilities; only compiled when `audit-replay`
  feature is on.
- `supergraph.rs` - structural filter (filter 1): reduces the merged graph and
  computes `ShiftPressure`.
