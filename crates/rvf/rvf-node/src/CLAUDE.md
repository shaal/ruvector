# rvf-node/src

Sole source dir.

## Files

- `lib.rs` — `#[napi]`-decorated wrappers around `rvf_runtime::RvfStore`. Provides JS-friendly `MetadataEntry`/`MetadataValue`/`FilterExpr`/`FilterValue`/`QueryOptions`/`RvfOptions`/`DistanceMetric` re-exports and async `insert` / `query` / `delete` / `compact` / `status`. `map_rvf_err` converts `RvfError` to `napi::Error`.
