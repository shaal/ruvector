# ruvector-robotics/src/perception

Higher-level perception built atop `../bridge` primitives.

## Files

- `mod.rs` — module decls.
- `config.rs` — `PerceptionConfig` (clustering radius, voxel size, thresholds).
- `clustering.rs` — point-cloud clustering (e.g. DBSCAN/euclidean) producing object proposals.
- `obstacle_detector.rs` — obstacle detection pipeline (filter → cluster → bounding-box).
- `scene_graph.rs` — symbolic scene-graph construction from object proposals.
- `sensor_fusion.rs` — fuse multiple sensor streams into a single perception view.
