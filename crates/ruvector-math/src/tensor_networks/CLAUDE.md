# ruvector-math/src/tensor_networks

Compressed representations of high-dimensional tensors via network decompositions.

- `mod.rs` — re-exports `TensorNetwork`, `TensorNode`, `NetworkContraction`, `CPDecomposition`, `CPConfig`, plus TT and Tucker decompositions.
- `tensor_train.rs` — Tensor-Train decomposition.
- `tucker.rs` — Tucker decomposition (core tensor + factor matrices).
- `cp_decomposition.rs` — CP / CANDECOMP / PARAFAC decomposition (sum of rank-1 tensors).
- `contraction.rs` — `TensorNetwork`, `TensorNode`, `NetworkContraction` (general contraction graph).

Used for attention compression, quantum-inspired algorithms, and high-dimensional integration. See `../CLAUDE.md`.
