# ruQu/src

Source root for the `ruqu` crate (classical nervous system for quantum machines). Flat module layout — each file is a top-level module.

## Files

- `lib.rs` — Crate doc + module declarations + public re-exports.
- `adaptive.rs` — Adaptive thresholding for detector signals.
- `attention.rs` — Attention-based gating logic.
- `decoder.rs` — Syndrome decoder (consumes `fusion-blossom` when configured).
- `error.rs` — Error type.
- `fabric.rs` — 256-tile WASM fabric definitions.
- `filters.rs` — Signal-processing filters.
- `metrics.rs` — Counters and timing telemetry.
- `mincut.rs` — Dynamic min-cut coherence gate (El-Hayek/Henzinger/Li O(n^{o(1)}) algorithm).
- `parallel.rs` — Rayon parallel helpers.
- `schema.rs` — Wire/data schema definitions.
- `stim.rs` — Stim quantum simulator integration.
- `tile.rs` — Tile primitives (units of the 256-tile fabric).
- `traits.rs` — Public traits (`SyndromeBuffer`, etc.).
- `types.rs` — Public types (`DetectorBitmap`, `SyndromeRound`).
- `bin/` — Standalone binaries (see `bin/CLAUDE.md`).
