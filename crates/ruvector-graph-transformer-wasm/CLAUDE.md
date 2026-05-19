# ruvector-graph-transformer-wasm

WASM bindings for proof-gated graph attention (`JsGraphTransformer`). Provides sublinear attention over CSR-style graphs, Hamiltonian/spiking/causal/manifold/game-theoretic attention variants, verified training steps, and dimension proofs — all callable from the browser.

## Layout

- `Cargo.toml` — `cdylib` + `rlib`. Pure self-contained crate (no internal dep on a sibling `ruvector-graph-transformer` — implementation lives here). Aggressive release: `opt-level = "s"`, LTO, single codegen unit. Lints relaxed (research-tier).
- `src/lib.rs` — JS entry: `JsGraphTransformer` with `createProofGate`, `proveDimension`, `sublinearAttention`, `hamiltonianStep`, `spikingStep`, `causalAttention`, `productManifoldAttention`, `verifiedTrainingStep`, `gameTheoreticAttention`.
- `src/transformer.rs` — core transformer math (attention, proof gate, graph kernels).
- `src/utils.rs` — type conversions between `JsValue`, `Float64Array`, edge lists.
- `tests/web.rs` — `wasm-bindgen-test` integration tests (run with `wasm-pack test --headless --chrome`).
- `pkg/` — checked-in build output (npm-ready: `.js`, `.d.ts`, `.wasm`, `package.json`).

## Public API (JS)

`JsGraphTransformer` (constructor + ~10 graph/attention methods).

## Related

- `../ruvector-decompiler-wasm`, `../ruvector-domain-expansion-wasm`, `../rvf/rvf-wasm`, `../rvf/rvf-solver-wasm` — sibling WASM modules
