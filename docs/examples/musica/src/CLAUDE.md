# docs/examples/musica/src/

Rust source for the **Musica** example crate. Modules cover audio source separation, hearing-aid DSP, transcription, and visualization. This is a sibling-to-binary code tree under `docs/` and is not itself documentation.

## Key modules

- `lib.rs`, `main.rs` - library root and CLI entry point.
- Source separation: `separator.rs`, `enhanced_separator.rs`, `advanced_separator.rs`, `multi_res.rs`, `multitrack.rs`, `streaming_multi.rs`.
- DSP and front end: `stft.rs`, `phase.rs`, `lanczos.rs`, `audio_graph.rs`, `spatial.rs`.
- Learning/inference: `learned_weights.rs`, `neural_refine.rs`, `adaptive.rs`.
- Apps: `hearing_aid.rs`, `transcriber.rs`, `crowd.rs`, `real_audio.rs`, `visualizer.rs`.
- Evaluation: `benchmark.rs`, `evaluation.rs`, `musdb_compare.rs`.
- `hearmusica/` - hearing-aid block library (filters, compressor, limiter, mixer, presets).

## Related

- `../` - crate root with Cargo.toml, wasm/, scripts/.
- `hearmusica/` - dedicated subdir for DSP blocks.
