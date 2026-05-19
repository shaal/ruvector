# rvm/tests

Workspace member (`rvm-tests`, `publish = false`) holding cross-crate integration tests for the RVM hypervisor.

- `Cargo.toml` — pulls every RVM crate (`rvm-types`, `rvm-hal`, `rvm-cap`, `rvm-witness`, `rvm-proof`, `rvm-partition`, `rvm-sched`, `rvm-memory`, `rvm-coherence`, `rvm-boot`, `rvm-wasm`, `rvm-security`, `rvm-kernel`). Optional feature `ed25519 = ["rvm-proof/ed25519"]`.
- `src/lib.rs` — the integration test module tree.

Run with `cargo test -p rvm-tests` from the `crates/rvm/` directory. See `../CLAUDE.md`.
