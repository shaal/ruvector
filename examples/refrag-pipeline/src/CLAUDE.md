# refrag-pipeline/src

Library + two binaries implementing the REFRAG Compress-Sense-Expand pipeline.

## Important files
- `lib.rs` — re-exports the four pipeline stages.
- `types.rs` — entry/response types.
- `compress.rs` — `CompressionStrategy` (tensor compression).
- `sense.rs` — `PolicyNetwork` (retrieval-policy chooser).
- `expand.rs` — `ExpandLayer` (re-expansion to full tensor).
- `store.rs` — `RefragStoreBuilder` (vector store wiring atop `ruvector-core`).
- `main.rs` — `refrag-demo` binary.
- `benchmark.rs` — `refrag-benchmark` binary.

## Build
- From parent: `cargo build --release`.
