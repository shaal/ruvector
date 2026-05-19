Upstream `hnsw_rs` examples. Each is wired as a Cargo `[[example]]` in the parent `Cargo.toml`.

Files:
- `ann-glove25-angular.rs` - GloVe 25D angular ANN benchmark.
- `ann-mnist-784-euclidean.rs` - MNIST 784D euclidean ANN benchmark.
- `ann-sift1m-128-euclidean.rs` - SIFT1M 128D euclidean ANN benchmark.
- `levensthein.rs` - HNSW with custom Levenshtein distance.
- `random.rs` - smallest "hello world" example for sanity checks.
- `utils/` - shared helpers (HDF5 loader).

Run with `cargo run --release --example <name>` from the parent crate. HDF5 datasets must be downloaded separately (paths are hardcoded in each example).
