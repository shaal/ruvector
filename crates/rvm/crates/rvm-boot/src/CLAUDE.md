# rvm-boot/src

- `lib.rs` — crate root; declares modules and the public boot-sequence API.
- `entry.rs` — reset-vector entry point handed over by firmware (phase 0).
- `hal_init.rs` — HAL bring-up: timer, MMU, interrupts.
- `sequence.rs` — orchestrates the 7-phase (ADR-137) sequence with witness-gated transitions.
- `measured.rs` — measured-boot helpers (per-phase hash, optional via `sha2`).

See `../CLAUDE.md`.
