# ruvix-dtb/src

## Files

- `lib.rs` — crate root; re-exports `DeviceTree` and `DtbError`.
- `header.rs` — FDT header struct + magic/version validation.
- `parser.rs` — main parser entry (`DeviceTree::parse`).
- `node.rs` — node iteration / navigation.
- `property.rs` — property iteration + typed accessors.
- `error.rs` — `DtbError` enum.
