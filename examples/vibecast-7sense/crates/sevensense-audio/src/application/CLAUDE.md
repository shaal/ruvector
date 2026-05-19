# sevensense-audio/src/application

Application layer of the audio bounded context.

## Files
- `mod.rs` - Re-exports.
- `services.rs` - `AudioIngestionService` orchestrating file readers, resamplers, and segmenters to turn raw files into `CallSegment` entities.
- `error.rs` - Application-level errors.

## Related
- Domain types: `../domain/`.
- Concrete adapters: `../infrastructure/`.
