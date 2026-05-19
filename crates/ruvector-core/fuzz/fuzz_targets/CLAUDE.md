# ruvector-core/fuzz/fuzz_targets

Fuzz target binaries (each is a `libfuzzer-sys` `fuzz_target!` entry point).

- `fuzz_distance.rs` — fuzzes the distance kernels (`distance.rs`, `simd_intrinsics.rs`) against malformed / extreme inputs to catch UB or panics.
