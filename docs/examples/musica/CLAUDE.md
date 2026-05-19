# docs/examples/musica/

**Musica** - a full Rust example crate built on ruvector for audio source separation, hearing-aid DSP, transcription, and visualization. Despite living under `docs/`, this is a runnable Cargo project, not documentation.

## Top-level

- `Cargo.toml`, `Cargo.lock` - crate manifest.
- `src/` - Rust source (separation, transcription, DSP, visualization).
- `wasm/` - WASM browser build artifacts.
- `scripts/` - utility shell scripts (WASM check, test-audio download).
- `test_audio/` - audio fixtures (currently empty / downloaded on demand).

## Quick orientation

The interesting code lives in `src/lib.rs` and the per-feature modules (`separator.rs`, `enhanced_separator.rs`, `transcriber.rs`, `visualizer.rs`, `hearmusica/`). Run `scripts/check_wasm.sh` to validate the WASM build.

## Related

- `../` - sibling examples directory (BTSP, SQL).
- `../../guides/wasm-build-guide.md` - general WASM build guidance.
