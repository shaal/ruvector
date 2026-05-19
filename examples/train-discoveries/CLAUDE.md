# train-discoveries

`train-discoveries` example: a cross-domain discovery ETL pipeline built on `ruvector-core` (parallel) and `ruvector-solver` (forward-push, neumann). Trains discovery models over multi-domain inputs and emits JSON results.

## Files

- `Cargo.toml` - Manifest; depends on `ruvector-core`, `ruvector-solver`, serde, rand, tracing.
- `src/main.rs` (~21 KB) - End-to-end ETL + training driver.

## How to run

```bash
cargo run -p train-discoveries --release
```

## Tech stack

- Rust 2021. Internal: `ruvector-core` (parallel), `ruvector-solver` (forward-push, neumann).
- Aux: serde, serde_json, rand, tracing.

## Related

- Output corpus to learn from / compare with: `examples/data/discoveries/`.
- Discovery framework: `examples/data/framework/`.
- Sublinear solver: `crates/ruvector-solver`.
