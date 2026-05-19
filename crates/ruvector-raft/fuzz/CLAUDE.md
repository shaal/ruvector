# ruvector-raft/fuzz

`cargo-fuzz` harness for the Raft RPC layer.

- `Cargo.toml` / `Cargo.lock` — fuzz crate manifest.
- `fuzz_targets/` — fuzz target binaries.

Run with `cargo +nightly fuzz run fuzz_raft_messages` from this directory.
