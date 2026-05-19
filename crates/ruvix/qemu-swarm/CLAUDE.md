# ruvix-qemu-swarm

QEMU swarm simulation for distributed RuVix cluster testing. Boots multiple QEMU virt machines and orchestrates a logical
cluster (single-node, 3-node, 8-node configurations provided) so distributed protocols and replication can be exercised on real
kernel binaries.

## Files

- `Cargo.toml` — depends on `ruvix-types`, optional `ruvix-nucleus`. Async via tokio (full); serde + toml for config.
- `configs/` — TOML cluster configs (`single-node`, `3-node-cluster`, `8-node-swarm`).
- `scripts/` — shell helpers (`launch-swarm.sh`, `monitor.sh`).
- `src/` — see `src/CLAUDE.md`.

## Features

- `default = ["std"]`. `std` enables `ruvix-types/std` + `ruvix-nucleus/std`.
