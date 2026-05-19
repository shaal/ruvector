# ruvector-bench/tests

Integration tests that double as micro-benchmarks for the WASM cognitive stack.

## Files

- `wasm_stack_bench.rs` — measures container tick (<200 µs), SCS recompute (<5 ms / 500 vertices), canonical min-cut (<1 ms / 100 vertices), witness fragment (<50 µs / 64 vertices). Uses `ruvector_mincut::canonical::CactusGraph` and friends. Run with `cargo test --test wasm_stack_bench --release -- --nocapture`.
