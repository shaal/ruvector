# ruvector-collections/src

Flat-file source layout.

## Files

- `lib.rs` — crate doc + module declarations + warn(missing_docs).
- `manager.rs` — `CollectionManager` (DashMap-backed, alias support, persistence).
- `collection.rs` — `Collection`, `CollectionConfig`.
- `error.rs` — crate errors.
- `primality.rs` — public deterministic Miller-Rabin + tabled fast paths (ADR-151).
- `primality_kernel.rs` — inner kernel used by `primality.rs`; tables provided by `build.rs`.
