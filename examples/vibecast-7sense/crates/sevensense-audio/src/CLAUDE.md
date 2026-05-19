# sevensense-audio/src

Source for the audio bounded context.

## Files
- `lib.rs` - Crate root. Documents the DDD layering (domain, application, infrastructure) and re-exports.
- `spectrogram.rs` - Spectrogram generation (FFT, optional mel scaling) used by both ingestion and embedding.

## Subdirectories
- `domain/` - Recording / CallSegment entities and repository traits.
- `application/` - `AudioIngestionService` and use cases.
- `infrastructure/` - Symphonia file reader, Rubato resampler, energy segmenter.

## Related
- Parent: `../CLAUDE.md`.
