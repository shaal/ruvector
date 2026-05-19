# ruvix-fs

Minimal filesystem abstraction for the RuVix Cognition Kernel (ADR-087 Phase E). VFS layer with pluggable filesystem backends.

## Core abstractions

- `BlockDevice` — hardware abstraction for block-level I/O.
- `FileSystem` — implementation (FAT32 read-only, RamFS read-write).
- `Inode` — file/directory with read/write ops.
- `VfsMountPoint` / `VfsMountTable` — mount-point management.

## Files

- `Cargo.toml` — depends only on `ruvix-types`. Dev: proptest, criterion.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/fs_bench.rs` — VFS / FAT32 throughput.
- `tests/fs_test.rs` — integration tests.

## Features

- `default = ["alloc"]`. Other: `std`, `lfn` (FAT32 long-filenames), `fat32-write` (Phase 2), `stats`.
