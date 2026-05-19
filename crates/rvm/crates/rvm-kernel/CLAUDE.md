# rvm-kernel

Top-level integration crate for the RVM coherence-native microhypervisor. Wires HAL + capability + witness + proof + partition + scheduler + memory + coherence + boot + WASM + security into a single API surface. Also emits the runnable `rvm` binary used by the `Makefile` (`make run` boots it in QEMU).

## Layout

- `Cargo.toml` — `rlib` + `[[bin]] name = "rvm" path = "src/main.rs"`. Pulls every other RVM crate.
- `src/lib.rs` — public kernel API: composes subsystems and exposes the integration surface.
- `src/main.rs` — `rvm` binary entry point (links the linker script `../../rvm.ld`, runs the boot sequence, hands off to the scheduler).

See `../CLAUDE.md` and the workspace overview in `../../CLAUDE.md`.
