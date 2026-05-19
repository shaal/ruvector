# rvf-integration-tests

Workspace integration/acceptance test crate (`publish = false`). Library is empty — all logic lives in `tests/`.

## Layout

- `Cargo.toml` — name `rvf-integration-tests`. Deps: every core RVF crate (`rvf-types`, `rvf-wire`, `rvf-manifest`, `rvf-index`, `rvf-quant`, `rvf-crypto`, `rvf-runtime`) + `rvf-adapter-rvlite`. Dev infra: `ed25519-dalek`, `rand`, `tempfile`.
- `src/lib.rs` — comment only; no code. Crate exists to host the `tests/` directory.
- `tests/` — ~30 integration files; see `tests/CLAUDE.md`.

## Related

- All `../../rvf-*` crates under test
- `../../rvf-adapters/rvlite` is the adapter used as a thin test client
