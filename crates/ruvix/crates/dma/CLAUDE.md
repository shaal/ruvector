# ruvix-dma

Hardware-agnostic DMA (Direct Memory Access) controller abstraction for the RuVix Cognition Kernel. Provides zero-copy data
transfers between memory regions and peripheral devices. No unsafe code in the public API, `no_std`, cache-coherent buffers,
scatter-gather support, platform-agnostic.

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-hal`. Dev: proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.

## Features

- `default = []`, `std`.
