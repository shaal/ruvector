# rvf-wire

Wire-format reader/writer for the RuVector Format. Zero-copy segment serialisation with binary encoding/decoding, varint encoding, delta coding, hash computation, tail scanning, and per-segment-type codecs.

## Layout

- `Cargo.toml` — features `default = ["std"]`. Deps: `rvf-types`, `xxhash-rust` (xxh3), `crc32c`, `sha3`, `subtle`.
- `src/lib.rs` — module decls + top-level re-exports.
- `src/reader.rs` — `read_segment`, `read_segment_header`, `validate_segment`.
- `src/writer.rs` — `write_segment`, `calculate_padded_size`.
- `src/tail_scan.rs` — `find_latest_manifest` (scan from EOF to find the most recent manifest).
- `src/varint.rs` — variable-length integer encoding.
- `src/delta.rs` — delta encoding for sorted id streams.
- `src/hash.rs` — xxh3 / SHAKE-256 wrappers.
- `src/vec_seg_codec.rs` — VEC_SEG (vector data) codec.
- `src/index_seg_codec.rs` — INDEX_SEG (HNSW layers) codec.
- `src/hot_seg_codec.rs` — hot-tier segment codec.
- `src/manifest_codec.rs` — manifest TLV codec (paired with `../rvf-manifest`).

## Public API

`read_segment`, `read_segment_header`, `validate_segment`, `write_segment`, `calculate_padded_size`, `find_latest_manifest`.

## Related

- `../rvf-types` — segment layout
- `../rvf-manifest` — uses `manifest_codec`
- `../rvf-runtime` — orchestrates wire reads/writes
