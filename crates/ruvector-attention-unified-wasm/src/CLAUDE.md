# ruvector-attention-unified-wasm/src

WASM glue mapping each family of attention mechanisms to `wasm-bindgen` types.

## Files

- `lib.rs` — crate documentation listing all 18+ mechanisms, module declarations, panic-hook init.
- `neural.rs` — neural attention bindings from `ruvector-attention` (Scaled Dot-Product, Multi-Head, Hyperbolic, Linear, Flash, Local-Global, MoE).
- `dag.rs` — DAG attention bindings from `ruvector-dag` (Topological, Causal Cone, Critical Path, MinCut-Gated, Hierarchical Lorentz, Parallel Branch, Temporal BTSP).
- `graph.rs` — GNN/graph attention bindings from `ruvector-gnn` (GAT, GCN, GraphSAGE).
- `mamba.rs` — Mamba selective state space model bindings.

Each module exposes `#[wasm_bindgen]` structs/functions; serialization uses `serde-wasm-bindgen`.
