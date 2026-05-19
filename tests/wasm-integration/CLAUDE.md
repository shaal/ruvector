`wasm-bindgen-test` suites for the edge-net WASM crates. Designed to run in both Node.js and browser environments.

Files:
- `mod.rs` - module declarations plus a `common` submodule of test helpers: `random_vector`, `assert_vectors_approx_eq`, `assert_finite`, `assert_in_range`, `create_test_attention_pattern`, `softmax`, `cosine_similarity`.
- `attention_unified_tests.rs` - `ruvector-attention-unified-wasm`: multi-head attention, Mamba SSM, etc.
- `learning_tests.rs` - `ruvector-learning-wasm`: MicroLoRA, SONA adaptive learning.
- `nervous_system_tests.rs` - `ruvector-nervous-system-wasm`: bio-inspired neural components.
- `economy_tests.rs` - `ruvector-economy-wasm`: economic coordination mechanisms.
- `exotic_tests.rs` - `ruvector-exotic-wasm`: NAOs, morphogenetic networks, time crystals.

Run with `wasm-pack test --node` (or `--chrome`/`--firefox`) from the relevant crate directory under `../../crates/`. Helpers in `common` are exposed for reuse by each test module.
