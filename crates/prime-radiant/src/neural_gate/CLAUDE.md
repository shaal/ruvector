# prime-radiant/src/neural_gate

Neural-gate decision module: takes encoded coherence features and emits a gate decision (allow / refuse / escalate) using a small neural classifier.

## Files

- `mod.rs` — module entry.
- `config.rs` — neural-gate config (architecture, thresholds).
- `encoding.rs` — feature encoder from coherence outputs to gate input vector.
- `gate.rs` — gate forward pass.
- `decision.rs` — `GateDecision` value object.
- `error.rs` — module errors.
