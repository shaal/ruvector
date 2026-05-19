# cognitum-gate-kernel/src

Core no_std kernel for one coherence-gate worker tile. Compiles to both `cdylib` (WASM) and `rlib`.

## Files

- `lib.rs` — crate root; `#![cfg_attr(not(feature = "std"), no_std)]`; sets up alloc/allocator for WASM, re-exports `TileState`, `Delta`, `Report`, `WitnessFragment` and the three WASM exports (`ingest_delta`, `tick`, `get_witness_fragment`).
- `shard.rs` — `CompactGraph`: vertices, edges, packed adjacency. Largest budget chunk (~42 KB).
- `delta.rs` — `Delta` enum: edge add/remove/weight update; helper constructors (`Delta::edge_add(u, v, w)`).
- `evidence.rs` — `EvidenceAccumulator`: sequential e-value accumulation, sliding window of hypotheses.
- `report.rs` — `Report` struct rolled up per `tick`, includes graph stats and evidence summary.
- `canonical_witness.rs` — pseudo-deterministic witness fragment generator (feature `canonical-witness`).

## Memory budget

| Component | Size |
|-----------|------|
| CompactGraph | ~42 KB |
| EvidenceAccumulator | ~2 KB |
| TileState | ~1 KB |
| Stack/control | ~19 KB |

## Public API surface

`TileState::new(tile_id: u32)`; `ingest_delta(&Delta)`; `tick(epoch: u64) -> Report`; `get_witness_fragment() -> WitnessFragment`.

See parent `../CLAUDE.md`.
