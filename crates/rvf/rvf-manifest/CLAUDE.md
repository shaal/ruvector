# rvf-manifest

Two-level manifest system for the RuVector Format, enabling progressive boot:

- **Level 0** — fixed 4096-byte block at EOF with hotset pointers for instant query
- **Level 1** — variable-size TLV records giving the full segment directory

A reader only needs Level 0 to start serving approximate queries; Level 1 loads asynchronously. `no_std` compatible.

## Layout

- `Cargo.toml` — features `default = ["std"]`. Deps: `rvf-types`, `crc32c`. Dev: `tempfile`.
- `src/lib.rs` — module decls + public re-exports.
- `src/level0.rs` — `read_level0`, `validate_level0`, `write_level0`.
- `src/level1.rs` — `Level1Manifest`, `ManifestTag`, `TlvRecord`, `read_tlv_records`, `write_tlv_records`.
- `src/boot.rs` — `boot_phase1`/`boot_phase2`, `extract_hotset_offsets`, `BootState`, `HotsetPointers`.
- `src/directory.rs` — `SegmentDirEntry`, `SegmentDirectory`.
- `src/chain.rs` — `OverlayChain` (parent → child manifest chaining).
- `src/writer.rs` — orchestrated manifest write.

## Public API

L0/L1 read/write helpers, boot APIs, `SegmentDirectory`, `OverlayChain`, `HotsetPointers`.

## Related

- `../rvf-types` — manifest-related types
- `../rvf-wire::manifest_codec` — codec used internally
- `../rvf-runtime` — boots stores via this crate
