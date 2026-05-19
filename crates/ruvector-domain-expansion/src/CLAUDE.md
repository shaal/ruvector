# ruvector-domain-expansion/src

Source layout for the cross-domain transfer-learning engine.

## Files

- `lib.rs` — crate docs (two-layer architecture: policy layer over operator layer), module decls, public re-exports.
- `domain.rs` — `DomainId`, `Domain` trait, `Task`, `Solution`, `Evaluation`.
- `meta_learning.rs` — `MetaThompsonEngine` with `DecayingBeta` posteriors keyed by `(ArmId, ContextBucket)`.
- `policy_kernel.rs` — `PolicyKnobs`, `PopulationSearch` (variant population, mutate, top-k retention).
- `planning.rs` — structured-planning domain implementation (steps + deps + resources).
- `rust_synthesis.rs` — Rust-program-synthesis domain (spec → function).
- `tool_orchestration.rs` — multi-tool agent coordination domain.
- `transfer.rs` — `TransferPrior`, prior dampening, transfer-protocol orchestration.
- `cost_curve.rs` — convergence/regret/pareto utilities used to score acceleration.
- `rvf_bridge.rs` — `#[cfg(feature = "rvf")]` segment serialisation.
- `error.rs` — `Error` type via `thiserror`.
