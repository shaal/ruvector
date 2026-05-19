# docs/examples/musica/src/hearmusica/

DSP block library for the **hearmusica** hearing-aid app within Musica. Each file implements one audio processing block.

## Modules

- `mod.rs` - module aggregator.
- Building blocks: `block.rs`, `gain.rs`, `filter.rs`, `delay.rs`, `feedback.rs`, `mixer.rs`.
- Dynamics: `compressor.rs`, `limiter.rs`.
- Application: `separator_block.rs` (wraps source separation as a block), `presets.rs` (preset configurations).

## Related

- `../hearing_aid.rs` - the hearing-aid app that composes these blocks.
- `../audio_graph.rs` - audio graph runtime.
