# cognitum-gate-tilezero

Native arbiter for TileZero in the Anytime-Valid Coherence Gate. TileZero is the
central authority in a 256-tile WASM fabric: it merges worker tile reports into a
supergraph, makes global gate decisions (Permit/Defer/Deny), issues Ed25519-signed
permit tokens, and maintains a hash-chained witness receipt log.

## Important files
- `Cargo.toml` - crate manifest. Features: `mincut` (enables `ruvector-mincut`),
  `audit-replay`. Bench `decision_bench` (criterion, async-tokio).
- `src/lib.rs` - public re-exports + shared types (`ActionId`, `VertexId`, `EdgeId`,
  `TileId`).
- `src/decision.rs` - `GateDecision`, `GateThresholds`, `DecisionFilter`,
  `ThreeFilterDecision`, `EvidenceDecision`, `DecisionOutcome`.
- `src/evidence.rs` - evidence aggregation (`AggregatedEvidence`, `EvidenceFilter`).
- `src/merge.rs` - `ReportMerger`, `MergeStrategy`, `MergedReport`, `WorkerReport`.
- `src/permit.rs` - permit token issuance/verification (`PermitToken`, `PermitState`,
  `Verifier`, `TokenDecodeError`, `VerifyError`).
- `src/receipt.rs` - hash-chained audit log (`ReceiptLog`, `WitnessReceipt`,
  `WitnessSummary`, `TimestampProof`).
- `src/replay.rs` - replay verification (gated by `audit-replay`).
- `src/supergraph.rs` - reduced graph / structural filter / shift pressure.

## Public API surface
Three-filter gate pipeline (structural -> evidence -> decision) -> signed
`PermitToken` + receipt entry. Async with `tokio::sync::RwLock`. Crypto via
`blake3` + `ed25519-dalek`.

## Tests / benches / examples
- `tests/` - decision, merge, permit, receipt unit tests.
- `tests_disabled/replay_tests.rs` - parked replay tests.
- `benches/` - `decision_bench`, `merge_bench`, `crypto_bench`, `benchmarks`.
- `examples/` - `basic_gate.rs`, `human_escalation.rs`, `receipt_audit.rs`.

## Related
- Sibling `cognitum-gate-kernel` (worker tile side).
- `neural-trader-coherence` reuses gate concepts for trading regime decisions.
