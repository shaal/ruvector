# rvf-import

Migration tools for importing JSON, CSV/TSV, and NumPy `.npy` data into RVF stores. Ships a library + a single `rvf-import` binary.

## Layout

- `Cargo.toml` — `[[bin]] name = "rvf-import" path = "src/bin/rvf_import.rs"`. Deps: `rvf-runtime`, `rvf-types` (`std`), `serde`/`serde_json`, `clap` (derive), `csv`.
- `src/lib.rs` — `VectorRecord { id, vector, metadata }` + module decls.
- `src/json.rs` — JSON importer.
- `src/csv_import.rs` — CSV / TSV importer.
- `src/numpy.rs` — NumPy `.npy` importer.
- `src/progress.rs` — progress reporting helpers (counters, ETA).
- `src/bin/rvf_import.rs` — clap-driven binary calling the importers above.

## Public API

`VectorRecord`, JSON/CSV/NumPy importer functions, progress utilities.

## Related

- `../rvf-runtime` — destination store API (`RvfStore`, `RvfOptions`, `MetadataEntry`)
- `../rvf-cli` — alternative CLI for general RVF ops
