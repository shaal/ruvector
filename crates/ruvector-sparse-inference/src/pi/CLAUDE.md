# ruvector-sparse-inference/src/pi

pi-based structural primitives used for calibration and drift detection. Pi's
irrational / non-repeating structure avoids power-of-2 resonance artefacts in
quantization.

- `mod.rs` — `PiContext` and module roots.
- `constants.rs` — pi-derived constant tables.
- `angular.rs` — hyperspherical / angular embeddings using pi phase encoding.
- `chaos.rs` — deterministic pseudo-randomness without RNG state.
- `drift.rs` — quantization drift detection via pi transforms.
