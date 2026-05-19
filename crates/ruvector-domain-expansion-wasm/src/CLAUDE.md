# ruvector-domain-expansion-wasm/src

Sole source dir.

## Files

- `lib.rs` — `WasmDomainExpansionEngine` wraps `DomainExpansionEngine`. Exposes `domainIds`, task/solution/evaluation flow, transfer priors, population search, and the acceleration scoreboard to JS via `wasm-bindgen` and `serde-wasm-bindgen`.
