# rvf-index

Progressive HNSW indexing for RVF. Implements the three-layer model:

- **Layer A**: entry points + coarse routing (<5 ms load, ~0.70 recall)
- **Layer B**: partial adjacency for hot region (100 ms–1 s load, ~0.85 recall)
- **Layer C**: full HNSW adjacency (seconds to load, ≥0.95 recall)

`no_std` compatible (alloc required). Currently defines its own local types — `rvf-types` integration is deferred until that crate stabilises.

## Layout

- `Cargo.toml` — features `default = ["std"]`, `std`, `simd`. Dev-dep: `rand`.
- `src/lib.rs` — module decls + public re-exports.
- `src/hnsw.rs` — `HnswConfig`, `HnswGraph`, `HnswLayer`.
- `src/layers.rs` — `IndexLayer`, `IndexState`, `LayerA`, `LayerB`, `LayerC`, `PartitionEntry`.
- `src/builder.rs` — `build_layer_a/b/c`, `build_full_index`.
- `src/codec.rs` — `encode_index_seg`, `decode_index_seg`, `IndexSegData`, `IndexSegHeader`, `CodecError`.
- `src/distance.rs` — `cosine_distance`, `dot_product`, `l2_distance`.
- `src/progressive.rs` — progressive load/promote logic across layers.
- `src/traits.rs` — abstract traits for distance + index interfaces.

## Public API

`HnswConfig`/`HnswGraph`/`HnswLayer`, `LayerA`/`LayerB`/`LayerC`, builder functions, codec types, distance functions.

## Related

- `../rvf-runtime` — wires the index into the store
- `../rvf-wire::index_seg_codec` — segment-level encoding companion
