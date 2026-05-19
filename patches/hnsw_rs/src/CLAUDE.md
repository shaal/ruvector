Library sources for the vendored `hnsw_rs` crate.

Files:
- `lib.rs` - module wiring and the lazy_static `env_logger` initializer.
- `prelude.rs` - convenience re-exports (`api`, `hnsw`, `filter`, `hnswio`, plus `anndists::dist::distances::*`).
- `api.rs` - public types and traits exposed to library users.
- `hnsw.rs` - core HNSW graph implementation (~76KB). Defines `Hnsw<T, D>` plus search/insert/level logic and `modify_level_scale` for recall-vs-cpu tuning.
- `hnswio.rs` - serialization, dump/reload, and mmap-aware IO (~66KB).
- `libext.rs` - C FFI extern interface for use from other languages (~42KB).
- `datamap.rs` - mmap-backed data file management.
- `flatten.rs` - flattened in-memory layout helpers.
- `filter.rs` - search-time filter trait (small).

Distance functions live in the external `anndists` crate; SIMD acceleration is gated behind the `stdsimd` / `simdeez_f` features in the parent Cargo.toml.
