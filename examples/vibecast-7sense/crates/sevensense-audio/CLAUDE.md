# sevensense-audio

Audio bounded context for 7sense: decoding (WAV/FLAC/MP3/Ogg via Symphonia), resampling (Rubato), spectrogram generation, segment detection.

## Files
- `Cargo.toml` - Depends on `sevensense-core`, `tokio`, `async-trait`, plus audio deps (workspace `symphonia`, `rubato`, `hound`).
- `src/lib.rs` - Crate root and architecture overview.
- `src/spectrogram.rs` - Spectrogram generation utilities (FFT, mel scaling).
- `src/domain/` - Domain entities (Recording, CallSegment), repository traits.
- `src/application/` - `AudioIngestionService` and friends.
- `src/infrastructure/` - File reader (Symphonia), resampler (Rubato), energy-based segmenter.
- `benches/spectrogram_bench.rs` - Criterion benchmark for spectrogram throughput.

## Build / bench
```
cargo build -p sevensense-audio
cargo bench -p sevensense-audio
```

## Related
- Embeddings consume segments: `../sevensense-embedding/`.
- Workspace benches: `../../benches/`.
