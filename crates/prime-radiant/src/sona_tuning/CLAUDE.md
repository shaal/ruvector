# prime-radiant/src/sona_tuning

Self-optimizing threshold tuning via `ruvector-sona` (EWC++ + MicroLoRA + ReasoningBank).

## Files

- `mod.rs` — module entry.
- `config.rs` — tuner config (learning rate, replay buffer size).
- `tuner.rs` — `SonaTuner`: drives the EWC++ updates against observed coherence outcomes.
- `adjustment.rs` — applies tuned adjustments back into engine thresholds.
- `error.rs` — module errors.

## Related

- `crates/sona` (ruvector-sona) — underlying SonaEngine, EwcPlusPlus, ReasoningBank.
