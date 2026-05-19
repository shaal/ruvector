# sevensense-core/src

Source for the foundational `sevensense-core` crate.

## Files
- `lib.rs` - Crate root. Module declarations + public re-exports.
- `config.rs` - Shared configuration types.
- `error.rs` - Cross-crate error types (re-exported from `lib.rs`).
- `identifiers.rs` - Strongly-typed entity ID newtypes.
- `values.rs` - Value objects (`GeoLocation`, `Timestamp`, `AudioMetadata`, ...).
- `traits.rs` - Shared traits.
- `telemetry.rs` - OpenTelemetry / tracing wiring.

## Subdirectories
- `domain/` - Domain entities, errors, and events.

## Related
- Parent: `../CLAUDE.md`.
