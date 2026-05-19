# neural-trader-replay

Witnessable replay segments, RVF serialization, and audit receipt logging for the RuVector Neural Trader (ADR-084). Provides sealed, signed market-event windows with coherence statistics and tamper-detection hashes.

## Layout

- `Cargo.toml` — `publish = false`; deps on `neural-trader-core` and `neural-trader-coherence`, plus `serde`/`serde_json`/`anyhow`.
- `src/lib.rs` — single-file crate; all public types live here.

## Public API surface

- `ReplaySegment` — sealed segment with `segment_id`, `symbol_id`, time bounds, events, embedding snapshot, labels, coherence stats, lineage, and a 16-byte witness hash.
- `SegmentKind` — enum: `HighUncertainty`, `LargeImpact`, `RegimeTransition`, `StructuralAnomaly`, `RareQueuePattern`, `HeadDisagreement`, `Routine`.
- `CoherenceStats`, `SegmentLineage` — companion value objects.
- Re-exports `CoherenceDecision`, `RegimeLabel`, `WitnessReceipt` (from `neural-trader-coherence`) and `MarketEvent` (from `neural-trader-core`).

## Tests / examples / benches

None inside this crate.

## Related crates

- `crates/neural-trader-core` — `MarketEvent` source-of-truth types.
- `crates/neural-trader-coherence` — `WitnessReceipt` and coherence decision types.
- See also: any sibling `neural-trader-*` crate for the broader trader stack.
