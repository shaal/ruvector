# rvf-import/src

Source.

## Files

- `lib.rs` — `VectorRecord { id: u64, vector: Vec<f32>, metadata: Vec<MetadataEntry> }` plus module decls.
- `json.rs` — JSON importer.
- `csv_import.rs` — CSV/TSV importer (uses `csv` crate).
- `numpy.rs` — NumPy `.npy` importer.
- `progress.rs` — progress counters / ETA formatting.
- `bin/rvf_import.rs` — clap binary; see `bin/CLAUDE.md`.
