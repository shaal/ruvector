# prime-radiant/src/tiles

256-tile coherence fabric integration via `cognitum-gate-kernel`. Tiles let the engine partition very large substrates into evidence-accumulating WASM-friendly chunks.

## Files

- `mod.rs` — module entry.
- `adapter.rs` — bridges `TileState`, `Delta`, `WitnessFragment`, `EvidenceAccumulator` into prime-radiant.
- `fabric.rs` — full 256-tile fabric driver.
- `coordinator.rs` — coordinates per-tile updates and aggregation.
- `error.rs` — module errors.

## Related

- `crates/cognitum-gate-kernel` — the 256-tile WASM coherence fabric.
