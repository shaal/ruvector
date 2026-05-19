# ruvector-robotics/benches

Criterion benchmarks for the robotics platform.

## Files

- `robotics_benchmarks.rs` — covers (1) point-cloud conversion, (2) spatial brute-force kNN via `SpatialIndex`, (3) obstacle-detection pipeline, (4) trajectory prediction (linear, polynomial), (5) spatial-softmax attention, (6) behavior-tree tick, (7) scene-graph construction. Registered as `[[bench]] name = "robotics_benchmarks"` in `../Cargo.toml`.
