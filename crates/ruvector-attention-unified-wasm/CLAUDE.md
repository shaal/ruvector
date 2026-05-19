# ruvector-attention-unified-wasm

Unified WebAssembly bindings for 18+ attention mechanisms (Neural, DAG, Graph/GNN, Mamba SSM). Wraps `ruvector-attention`, `ruvector-dag`, and `ruvector-gnn` (all `default-features=false, features=["wasm"]`) behind one `wasm-bindgen` surface for JS/TS consumers.

## Important files

- `Cargo.toml` — `cdylib + rlib`. Pulls `ruvector-attention`, `ruvector-dag`, `ruvector-gnn` (wasm features) and the `wasm-bindgen` / `js-sys` / `web-sys` toolchain.
- `src/lib.rs` — crate doc enumerating all attention variants; module declarations and shared utilities.
- `src/neural.rs` — wraps `ruvector-attention`: Scaled Dot-Product, Multi-Head, Hyperbolic, Linear (Performer), Flash, Local-Global, MoE.
- `src/dag.rs` — wraps `ruvector-dag`: Topological, Causal Cone, Critical Path, MinCut-Gated, Hierarchical Lorentz, Parallel Branch, Temporal BTSP.
- `src/graph.rs` — wraps `ruvector-gnn`: GAT, GCN, GraphSAGE.
- `src/mamba.rs` — wraps Mamba selective state space model attention.
- `pkg/` — `wasm-pack` build output (gitignored content / artifacts).

## Public API surface

Exposes JS classes/functions tagged with `#[wasm_bindgen]` named per attention mechanism. Consumers initialize via `await init()` and instantiate the class for the desired mechanism.

## Related

- Upstream Rust crates: `crates/ruvector-attention`, `crates/ruvector-dag`, `crates/ruvector-gnn`.
- Companion narrower WASM crates: `crates/ruvector-mincut-wasm`, `crates/ruvector-gnn-wasm`.
