# ruvector-diskann-node/src

NAPI-RS binding surface for `ruvector-diskann`.

## Files

- `lib.rs` — sole source file. Defines:
  - `#[napi(object)] DiskAnnOptions` — JS-facing config (dim, max_degree, build_beam, search_beam, alpha, PQ params, storage_path).
  - `#[napi(object)] DiskAnnSearchResult` — `{ id: String, distance: f64 }`.
  - `#[napi] DiskAnn` — wraps `Arc<RwLock<ruvector_diskann::DiskAnnIndex>>`; constructor maps `DiskAnnOptions` to
    `ruvector_diskann::DiskAnnConfig`.
