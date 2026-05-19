Vendored fork of `hnsw_rs` v0.3.3 (Malkov/Yashunin HNSW), wired into the workspace via `hnsw_rs = { path = "./patches/hnsw_rs" }` in the root Cargo.toml.

Layout (mirrors upstream):
- `Cargo.toml` - crate manifest. Depends on `anndists` for distance functions, `rayon`/`parking_lot` for parallelism, `mmap-rs` for mmapped data files. Optional features `stdsimd` (nightly) and `simdeez_f` (stable x86).
- `Changes.md` - upstream changelog (0.3.3, 0.3.2, 0.3.1, ...).
- `src/` - library sources (`hnsw.rs`, `hnswio.rs`, `libext.rs`, `datamap.rs`, `flatten.rs`, `filter.rs`, `api.rs`, `prelude.rs`, `lib.rs`).
- `examples/` - ANN demos (glove25-angular, mnist-784, sift1m-128, levenshtein, random) plus `utils/annhdf5.rs`.
- `tests/` - upstream test suite (deallocation, filter, serialize-parallel).

Build with `cargo build` inside this directory or `cargo build -p hnsw_rs` from the workspace. An identical mirror exists at `../../scripts/patches/hnsw_rs/`.
