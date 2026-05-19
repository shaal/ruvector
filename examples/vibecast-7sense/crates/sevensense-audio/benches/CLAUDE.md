# sevensense-audio/benches

Criterion benchmarks for the audio bounded context.

## Files
- `spectrogram_bench.rs` - Measures spectrogram generation throughput (FFT + mel scaling) on representative audio durations.

## Run
```
cargo bench -p sevensense-audio
```

## Related
- Source: `../src/spectrogram.rs`.
- Workspace-level benches: `../../../benches/`.
