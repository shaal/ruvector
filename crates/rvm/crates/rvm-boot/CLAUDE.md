# rvm-boot

Deterministic phased boot sequence for the RVM microhypervisor (ADR-137 and ADR-140). Each phase is gated by a witness entry and must complete before the next phase begins.

## Boot phases

ADR-137 (7-phase deterministic boot): reset vector → hardware detect → MMU setup → enter EL2 → kernel-object init → first witness (genesis attestation) → scheduler entry. ADR-140 (legacy alt sequence): HAL init → memory pool → cap table → witness trail → scheduler → root partition → handoff.

## Layout

- `Cargo.toml` — `rlib`; deps: `rvm-types`, `rvm-hal`, `rvm-partition`, `rvm-witness`, `rvm-sched`, `rvm-memory`; optional `sha2`; `subtle`.
- `src/lib.rs` — module wiring and phase orchestrator.
- `src/entry.rs` — reset-vector entry point.
- `src/hal_init.rs` — phase 1/0: HAL bring-up (timer, MMU, interrupts).
- `src/sequence.rs` — phase ordering, witness-gated transitions.
- `src/measured.rs` — measured-boot helpers (hash of each phase, optional `sha2`).

See `../CLAUDE.md` and the parent `../../CLAUDE.md`.
