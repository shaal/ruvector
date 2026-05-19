# sevensense-audio/src/infrastructure

Infrastructure adapters for the audio bounded context.

## Files
- `mod.rs` - Wires the adapters.
- `file_reader.rs` - Symphonia-backed file reader (WAV/FLAC/MP3/Ogg).
- `resampler.rs` - Rubato-backed resampler / normalizer.
- `segmenter.rs` - Energy-based call segmentation.

## Related
- Used by: `../application/services.rs`.
- Domain traits: `../domain/repository.rs`.
