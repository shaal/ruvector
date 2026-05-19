# ruvector-cognitive-container

Verifiable WASM cognitive container with canonical witness chains. Composes
cognitive primitives (graph ingest, min-cut, spectral analysis, evidence
accumulation) into a sealed container that produces a tamper-evident witness
chain linking every epoch to its predecessor. `crate-type = ["rlib"]` (consumed
by WASM hosts, not itself a cdylib).

## Important files
- `Cargo.toml` - workspace inheritance; minimal deps (`serde`, `serde_json`,
  `thiserror`). Dev: `proptest`.
- `src/lib.rs` - re-exports the public surface.
- `src/container.rs` - `CognitiveContainer`, `ContainerConfig`,
  `ContainerSnapshot`, `ComponentMask`, `Delta`, `TickResult`.
- `src/epoch.rs` - `EpochController`, `ContainerEpochBudget`, `Phase`.
- `src/error.rs` - `ContainerError`, `Result` alias.
- `src/memory.rs` - `Arena`, `MemoryConfig`, `MemorySlab` - deterministic
  bump allocator backing the container.
- `src/witness.rs` - `WitnessChain`, `ContainerWitnessReceipt`,
  `CoherenceDecision`, `VerificationResult`.

## Public API surface
Construct a `CognitiveContainer`, drive `tick`s with input deltas (producing a
`TickResult` + receipt), snapshot/restore state. Receipts form a hash chain
verifiable against a `WitnessChain` independent of execution.

## Tests
- `tests/container_bench.rs` - container throughput/integration test.

## Related
- Pairs with `cognitum-gate-tilezero` (whose `WitnessChain` concept this
  mirrors at container-scope).
- Likely consumed by WASM kernel hosts in `ruvector-wasm`.
