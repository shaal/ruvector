# ruvector-robotics/src/bridge

Foundational robotics primitives that other modules build on: point clouds, poses, spatial indexing, perception pipeline scaffolding.

## Files

- `mod.rs` — module declarations + re-exports.
- `config.rs` — `BridgeConfig` knobs for indexing and pipeline stages.
- `converters.rs` — sensor frame ↔ canonical robot frame conversions; format adapters (PCD/PLY/ROS-style payloads).
- `gaussian.rs` — Gaussian utilities for occupancy/uncertainty modelling.
- `indexing.rs` — `SpatialIndex` (brute-force kNN over 3-D points; foundation for perception/cognitive lookups).
- `pipeline.rs` — staged perception pipeline plumbing.
- `search.rs` — search/query helpers over the spatial index.
