Rust counterpart of the RVF JS smoke test in `../smoke-test.js`.

Files:
- `rvf_smoke_test.rs` - 15-step end-to-end RVF lifecycle test: create store (dim=128, cosine), ingest 100 random vectors, query top-10, verify ordering / valid distances (0.0..2.0 for cosine), close + reopen for persistence, delete + compact, derive a child store, validate independent queryability and segment listing on parent + child, cleanup.

Note: `DistanceMetric` is not persisted in the RVF manifest, so reopened stores assume the same metric was supplied at create time.
