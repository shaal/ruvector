# ruvix-fs/src

## Files

- `lib.rs` — crate root; re-exports `VfsMountTable`, `RamFs`, `Fat32Fs`, supporting types.
- `vfs.rs` — VFS layer: `VfsMountPoint`, mount/unmount, path resolution.
- `block.rs` — `BlockDevice` trait.
- `fat32.rs` — read-only FAT32 implementation (write support behind `fat32-write` feature).
- `ramfs.rs` — RamFS read-write filesystem.
- `path.rs` — path parsing / canonicalization.
- `error.rs` — fs error enum.
