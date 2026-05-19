# neural-trader-replay/src

Single-file crate: everything lives in `lib.rs`.

## Files

- `lib.rs` — defines `ReplaySegment`, `SegmentKind`, `CoherenceStats`, `SegmentLineage`, and helpers for sealing/signing market-event replay windows. Uses a `VecDeque` ring buffer internally.

## Notes

No submodules; adding new types should keep this single-file layout unless the crate grows substantially.
