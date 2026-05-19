# rvf-wire/src

Source for binary wire format.

## Files

- `lib.rs` — module decls + re-exports of reader/writer/tail-scan entry points.
- `reader.rs` — `read_segment`, `read_segment_header`, `validate_segment`.
- `writer.rs` — `write_segment`, `calculate_padded_size`.
- `tail_scan.rs` — `find_latest_manifest`: scan from EOF backward to locate latest manifest block.
- `varint.rs` — varint encoding/decoding.
- `delta.rs` — delta encoding for sorted-id streams.
- `hash.rs` — xxh3 and SHAKE-256 hash wrappers.
- `vec_seg_codec.rs` — VEC_SEG codec (vector data segments).
- `index_seg_codec.rs` — INDEX_SEG codec (HNSW layer segments).
- `hot_seg_codec.rs` — hot-tier segment codec.
- `manifest_codec.rs` — TLV manifest codec, paired with `../rvf-manifest`.
