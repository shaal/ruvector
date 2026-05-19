# benchmarks/src

Source tree for the `ruvector-benchmarks` crate. The crate is binary-only — all entry points live under `bin/`.

## Important files
- `bin/` — one `.rs` file per `[[bin]]` declared in `../Cargo.toml`.

## Notes
- There is no `lib.rs`; shared helpers (if any) live alongside each binary.
- See `../Cargo.toml` for the binary list and feature flags.
