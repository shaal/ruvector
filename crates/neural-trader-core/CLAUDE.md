# neural-trader-core

Canonical market event types, graph schema, and ingest traits for the RuVector
Neural Trader (ADR-084). Single-file crate: defines the wire-stable structs every
other neural-trader-* crate consumes. `publish = false` (workspace-internal).

## Important files
- `Cargo.toml` - tiny manifest. Deps: `anyhow`, `serde`. Dev: `serde_json`.
- `src/lib.rs` - all type definitions.

## Public API surface
- `MarketEvent` - 16-byte event_id, ns-precision timestamps, fixed-point price/qty
  (price * 1e8), venue/symbol IDs, optional hashed order/participant IDs, flags
  and sequence number.
- `EventType` - `repr(u8)` discriminant: `NewOrder`, `ModifyOrder`, plus other
  order/trade lifecycle states.
- `Side` - bid/ask side enum.
- Graph schema and ingest traits used by the downstream ingest pipeline.

## Tests / benches
None in this crate. Consumers (`neural-trader-wasm`, `neural-trader-coherence`,
`neural-trader-strategies`, `neural-trader-replay`, `neural-trader-rt`) test the
types in context.

## Related
- `neural-trader-coherence` - coherence gates over event streams.
- `neural-trader-replay` - reservoir replay memory.
- `neural-trader-strategies` - strategy implementations.
- `neural-trader-wasm` - WASM bindings re-exporting these types.
