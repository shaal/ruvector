# sona/src/loops

Three-tier temporal learning architecture.

## Files

- `mod.rs` — Re-exports `BackgroundLoop`, `LoopCoordinator`, `InstantLoop`.
- `instant.rs` — **Loop A (Instant)**: per-request trajectory recording + micro-LoRA updates (rank 1-2 for instant learning).
- `background.rs` — **Loop B (Background)**: hourly pattern extraction + base-LoRA updates.
- `coordinator.rs` — `LoopCoordinator` orchestrating the three loops (instant, background, and the deeper "Loop C" weekly EWC++ consolidation referenced in the crate doc).
