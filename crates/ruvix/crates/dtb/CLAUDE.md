# ruvix-dtb

Zero-copy parser for Flattened Device Tree (FDT) blobs. Discovers hardware configuration at boot time. `no_std`, no allocations,
safe parsing with all data validated before access.

## FDT structure (parsed)

- Header (magic, version, offsets)
- Memory reservation block
- Structure block (nodes + properties)
- Strings block (property names)

## Files

- `Cargo.toml` — depends only on `ruvix-types`. Dev: proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.

## Features

- `default = []`, `std`, `alloc`.

## Public API

`DeviceTree::parse(blob: &[u8]) -> Result<DeviceTree, DtbError>` and node/property iteration.
