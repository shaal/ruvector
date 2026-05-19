# ruvector-domain-expansion

Cross-domain transfer-learning engine: a two-layer architecture combining a Meta Thompson Sampling policy layer (Beta priors, context buckets, population-based search) with deterministic operator domains (Rust synthesis, structured planning, tool orchestration). The acceptance gate is: training on Domain 1 must accelerate Domain 2 convergence vs. Domain-2-alone.

## Layout

- `Cargo.toml` — `rlib` only. Optional `rvf` feature pulls in `rvf-types`/`rvf-wire`/`rvf-crypto` for segment serialisation. Dev-deps include `proptest` and `criterion`.
- `src/lib.rs` — top-level crate docs + module declarations. Re-exports the public API listed below.
- `src/domain.rs` — `DomainId`, `Task`, `Solution`, `Evaluation` traits and the `Domain` abstraction.
- `src/meta_learning.rs` — `MetaThompsonEngine`, `ContextBucket`, `ArmId`, decaying Beta posteriors.
- `src/policy_kernel.rs` — `PolicyKnobs`, `PopulationSearch`, variant tuning.
- `src/planning.rs` — structured-planning domain (multi-step with deps/resources).
- `src/rust_synthesis.rs` — Rust function synthesis-from-spec domain.
- `src/tool_orchestration.rs` — multi-tool/agent coordination domain.
- `src/transfer.rs` — `TransferPrior`, dampened-prior injection, transfer protocol.
- `src/cost_curve.rs` — `CostCurve`, `CostCurvePoint`, `ParetoFront`, `RegretTracker`, `PlateauDetector`, `ConvergenceThresholds`, `CuriosityBonus`.
- `src/rvf_bridge.rs` — RVF segment serialisation (gated on `rvf` feature).
- `src/error.rs` — crate-wide error type.
- `benches/domain_expansion_bench.rs` — Criterion benchmark for task gen, transfer, search.

## Public API surface

`DomainExpansionEngine`, `MetaThompsonEngine`, `PopulationSearch`, `AccelerationScoreboard`, `TransferPrior`, `PolicyKnobs`, `CostCurve`, `ParetoFront`, `PlateauDetector`, `CuriosityBonus`, plus `Task`, `Solution`, `Evaluation`, `DomainId`, `ArmId`, `ContextBucket`.

## Related

- `../ruvector-domain-expansion-wasm` — WASM bindings to this crate
- `../ruvector-robotics` — consumes this crate under its `domain-expansion` feature
- `../rvf/rvf-types`, `../rvf/rvf-wire`, `../rvf/rvf-crypto` — required by `rvf` feature
