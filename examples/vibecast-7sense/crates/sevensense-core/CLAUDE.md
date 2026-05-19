# sevensense-core

Foundational crate for the 7sense platform: strongly-typed identifiers, value objects (`GeoLocation`, `Timestamp`, `AudioMetadata`), shared error types, domain entities, and events. Depended on by every other `sevensense-*` crate.

## Files
- `Cargo.toml` - Library crate. Depends on `serde`, `serde_json`, `thiserror`, `uuid`, `chrono`, `async-trait`.
- `src/lib.rs` - Crate root; re-exports `error`, `identifiers`, `values`.
- `src/config.rs` - Shared configuration types.
- `src/error.rs` - Shared error enum (`SevenSenseError`).
- `src/identifiers.rs` - Strongly-typed entity IDs.
- `src/values.rs` - Value objects.
- `src/traits.rs` - Cross-cutting traits.
- `src/telemetry.rs` - Tracing / OpenTelemetry helpers.
- `src/domain/` - Domain entities, errors, events.

## Build
```
cargo build -p sevensense-core
```

## Related
- Used by every other `sevensense-*` crate.
