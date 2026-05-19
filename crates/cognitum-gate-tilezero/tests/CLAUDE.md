# cognitum-gate-tilezero/tests

Integration tests for the TileZero arbiter modules. Run with
`cargo test -p cognitum-gate-tilezero`.

## Files
- `decision_tests.rs` - three-filter pipeline outcomes against `GateThresholds`
  edge cases.
- `merge_tests.rs` - `ReportMerger` correctness across `MergeStrategy` variants
  and disagreeing workers.
- `permit_tests.rs` - Ed25519 token encode/decode/verify round trips and the
  `TokenDecodeError` / `VerifyError` paths.
- `receipt_tests.rs` - hash-chain integrity of `ReceiptLog`, tampering
  detection, and `WitnessSummary` aggregation.
