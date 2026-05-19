# ruvector-sparse-inference/src/sparse

Sparse FFN operator that skips cold neurons and only computes the hot subset
selected by the predictor.

- `mod.rs` — module roots and re-exports.
- `ffn.rs` — `SparseFfn` op, integrated with the precision/lanes subsystem.

Tested by `tests/unit/sparse_ffn_tests.rs`.
