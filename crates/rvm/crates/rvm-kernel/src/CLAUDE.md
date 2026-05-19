# rvm-kernel/src

- `lib.rs` — composes all RVM subsystems (HAL, cap, witness, proof, partition, sched, memory, coherence, boot, wasm, security) and exposes a unified kernel API.
- `main.rs` — `rvm` binary entry point. Linked with `../../rvm.ld`; on boot runs `rvm-boot` then enters the scheduler. Used by `make run` to launch QEMU.

See `../CLAUDE.md`.
