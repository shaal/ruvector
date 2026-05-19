# cognitum-gate-tilezero/examples

Runnable examples demonstrating TileZero arbiter usage. Each is a single-file
`cargo run --example <name>` binary.

## Files
- `basic_gate.rs` - minimal end-to-end gate: spin up TileZero, submit a few
  worker reports, observe Permit/Defer/Deny.
- `human_escalation.rs` - shows the Defer path: when filters disagree,
  decision escalates to a human reviewer pathway.
- `receipt_audit.rs` - emits a chain of `WitnessReceipt`s and walks the hash
  chain to verify integrity.
