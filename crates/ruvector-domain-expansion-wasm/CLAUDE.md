# ruvector-domain-expansion-wasm

WASM bindings for `../ruvector-domain-expansion`. Exposes the cross-domain transfer-learning engine, Meta Thompson Sampling, PolicyKernel population search, and the acceleration scoreboard to JavaScript. With the default `rvf` feature, also serialises priors/kernels/cost-curves into RVF wire segments.

## Layout

- `Cargo.toml` — `cdylib` + `rlib`. Default feature `rvf` forwards to `ruvector-domain-expansion/rvf`. Aggressive release profile: `opt-level = "z"`, LTO, single codegen unit, `panic = "abort"`, strip on.
- `src/lib.rs` — defines `WasmDomainExpansionEngine` and additional `#[wasm_bindgen]` wrappers around `DomainExpansionEngine`, `MetaThompsonEngine`, `PopulationSearch`, `AccelerationScoreboard`, etc.

## Public API (JS)

`WasmDomainExpansionEngine` (constructor, `domainIds()`, task generation, evaluation, transfer), plus thin wrappers over the engine types.

## Related

- `../ruvector-domain-expansion` — pure Rust source of truth
- `../rvf/rvf-types`, `../rvf/rvf-wire`, `../rvf/rvf-crypto` — RVF feature
