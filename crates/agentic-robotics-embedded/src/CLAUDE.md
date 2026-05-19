# agentic-robotics-embedded/src

Sole source dir for the embedded-targets crate.

## Files

- `lib.rs` — defines `EmbeddedPriority` (4-level enum) and `EmbeddedConfig` (tick_rate_hz, stack_size), plus a default-value unit test. No external runtime deps yet; Embassy/RTIC integration is gated behind the `embassy` / `rtic` features in Cargo.toml.
