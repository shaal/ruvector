# rvm-wasm

Optional WebAssembly guest runtime for RVM partitions. When enabled, a partition can host a Wasm module as an alternative to a native AArch64 / RISC-V / x86-64 guest. The module runs in a sandboxed interpreter; host functions are exposed via the capability system; every Wasm state transition is witness-logged. Agent lifecycle follows ADR-140 state machine. Per-partition resource quotas are enforced per epoch. Migration uses a 7-step protocol with DC-7 timeout.

This crate is compile-time optional; disabling it removes all Wasm code from the final binary.

`#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`.

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `rvm-partition`, `rvm-cap`, `rvm-witness`.
- `src/lib.rs` — crate root.
- `src/agent.rs` — agent lifecycle (ADR-140 state machine).
- `src/host_functions.rs` — capability-mediated host functions exposed to the Wasm guest.
- `src/quota.rs` — per-partition resource quotas (memory, time, IPC), enforced per epoch.
- `src/migration.rs` — 7-step Wasm-state migration with DC-7 timeout.

See `../CLAUDE.md`.
