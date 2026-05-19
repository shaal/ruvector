# ruvector-graph/fuzz

cargo-fuzz harness for the Cypher parser.

## Files

- `Cargo.toml` — Fuzz crate (separate package). Depends on `libfuzzer-sys` and the host `ruvector-graph` crate.
- `Cargo.lock` — Lockfile.
- `fuzz_targets/` — Individual fuzz binaries.

## Run

```
cargo +nightly fuzz run fuzz_cypher_parser
```
