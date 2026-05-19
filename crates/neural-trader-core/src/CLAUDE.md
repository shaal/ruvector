# neural-trader-core/src

Single-file source for the canonical neural-trader market event schema.

## Files
- `lib.rs` - all definitions: `MarketEvent` envelope, `EventType` discriminant
  (`NewOrder`, `ModifyOrder`, ...), `Side`, plus graph schema and ingest
  traits. Wire-stable - bump version if you change layout.
