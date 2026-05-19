# rvf-manifest/src

Source.

## Files

- `lib.rs` — `no_std` shim, module decls, public re-exports.
- `level0.rs` — fixed 4096-byte EOF block (hotset pointers). `read_level0`, `validate_level0`, `write_level0`.
- `level1.rs` — TLV records (`Level1Manifest`, `ManifestTag`, `TlvRecord`); full segment directory.
- `boot.rs` — `boot_phase1`/`boot_phase2`, `BootState`, `HotsetPointers`, `extract_hotset_offsets`.
- `directory.rs` — `SegmentDirEntry`, `SegmentDirectory` data model.
- `chain.rs` — `OverlayChain` for parent ↔ child manifest chains.
- `writer.rs` — orchestrated write path (L1 → L0 + CRC).
