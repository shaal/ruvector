Vendored, patched copies of upstream crates that the workspace consumes via path overrides in the root `Cargo.toml`.

Subdirectories:
- `hnsw_rs/` - vendored fork of jean-pierreBoth's `hnsw_rs` v0.3.3 (HNSW ANN library). Referenced from the workspace as `hnsw_rs = { path = "./patches/hnsw_rs" }`.

A duplicate copy lives at `../scripts/patches/hnsw_rs/` (artifact of script-driven patch generation; the workspace path dependency points at this directory, not the scripts copy).
