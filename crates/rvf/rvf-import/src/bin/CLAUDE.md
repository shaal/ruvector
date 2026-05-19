# rvf-import/src/bin

## Files

- `rvf_import.rs` → `rvf-import` binary. clap-driven entry that dispatches to the JSON/CSV/NumPy importers in the parent crate. Streams `VectorRecord`s into an `rvf_runtime::RvfStore`.
