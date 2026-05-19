# ruvector-cognitive-container/src

Source for the verifiable WASM cognitive container.

## Files
- `lib.rs` - module wiring and public re-exports.
- `container.rs` - core `CognitiveContainer` driver: `ContainerConfig`,
  `ContainerSnapshot`, `ComponentMask`, `Delta`, `TickResult`.
- `epoch.rs` - epoch lifecycle and budget enforcement (`EpochController`,
  `ContainerEpochBudget`, `Phase`).
- `error.rs` - `ContainerError` enum + `Result` alias (via `thiserror`).
- `memory.rs` - deterministic memory arena (`Arena`, `MemoryConfig`,
  `MemorySlab`) used to keep ticks reproducible.
- `witness.rs` - canonical witness chain: `WitnessChain`,
  `ContainerWitnessReceipt`, `CoherenceDecision`, `VerificationResult`.
