# ruvix-integration

Integration-test workspace member for the RuVix Cognition Kernel. `publish = false`. Holds the top-level ADR-087 acceptance suite
plus a single integration bench.

## Files

- `Cargo.toml` — depends on `ruvix-types`, `ruvix-region`, `ruvix-queue`, `ruvix-cap` (and likely more — see the Cargo.toml for
  the full list).
- `src/lib.rs` — shared test helpers.
- `tests/adr087_section17_acceptance.rs` — ADR-087 Section 17 acceptance criteria.
- `tests/syscall_flows.rs` — multi-syscall flow / interaction tests.
- `benches/integration_bench.rs` — cross-subsystem integration bench.
