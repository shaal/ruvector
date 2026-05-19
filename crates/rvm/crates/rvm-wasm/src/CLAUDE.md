# rvm-wasm/src

- `lib.rs` — crate root.
- `agent.rs` — agent lifecycle state machine (ADR-140).
- `host_functions.rs` — capability-gated host functions exposed to the Wasm guest.
- `quota.rs` — per-partition per-epoch resource quotas.
- `migration.rs` — 7-step migration protocol with DC-7 timeout.

See `../CLAUDE.md`.
