# rvf-index/src

Source.

## Files

- `lib.rs` — `no_std` shim, module decls, public re-exports.
- `hnsw.rs` — `HnswConfig` / `HnswGraph` / `HnswLayer` core HNSW data structures.
- `layers.rs` — three-tier `LayerA`/`LayerB`/`LayerC` plus `IndexLayer`, `IndexState`, `PartitionEntry`.
- `builder.rs` — `build_layer_a/b/c`, `build_full_index`.
- `codec.rs` — encode/decode `IndexSegData` + `IndexSegHeader`; `CodecError`.
- `distance.rs` — `cosine_distance`, `dot_product`, `l2_distance`.
- `progressive.rs` — promote/demote between layers; progressive load orchestration.
- `traits.rs` — abstract distance / index traits.
