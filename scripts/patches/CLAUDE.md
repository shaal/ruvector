Mirror of `../../patches/` containing the vendored `hnsw_rs` source. Appears to be a staging copy used by patch-generation tooling; the workspace path dependency in the root `Cargo.toml` resolves to `../../patches/hnsw_rs`, not this copy.

Subdirectories:
- `hnsw_rs/` - identical layout to `../../patches/hnsw_rs/`.

If you need to modify the patched crate, edit `../../patches/hnsw_rs/` (the actual workspace member) and then resync this mirror if a script depends on it.
