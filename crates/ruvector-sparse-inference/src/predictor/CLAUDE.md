# ruvector-sparse-inference/src/predictor

Low-rank neuron-activity predictor used to pick "hot" neurons before computing
the sparse FFN.

- `mod.rs` — predictor trait and re-exports.
- `lowrank.rs` — P*Q low-rank predictor for fast top-k neuron selection.

Tested by `tests/unit/predictor_tests.rs`.
