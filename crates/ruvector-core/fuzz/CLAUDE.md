# ruvector-core/fuzz

`cargo-fuzz` harness for ruvector-core. Run with `cargo +nightly fuzz run <target>`.

## Files

- `Cargo.toml` — fuzz crate manifest (separate from the parent).
- `Cargo.lock` — locked deps for reproducibility.
- `fuzz_targets/` — individual fuzz target binaries.
