# agentic-robotics-embedded

Embedded-systems support layer for the agentic-robotics framework targeting bare-metal/RTOS environments (Embassy, RTIC). Provides priority/config primitives that pair with `../agentic-robotics-core`.

## Layout

- `Cargo.toml` — package crate; depends on `agentic-robotics-core` only. Optional features `embassy` and `rtic` (deps currently commented out, pending integration).
- `src/lib.rs` — sole source file. Defines `EmbeddedPriority` (Low/Normal/High/Critical) and `EmbeddedConfig { tick_rate_hz, stack_size }`. Includes a unit test for default config.

## Public API

- `EmbeddedPriority` enum
- `EmbeddedConfig` struct (Default impl: 1000 Hz tick, 4096-byte stack)

## Related

- `../agentic-robotics-core` — main runtime types (Publisher/Subscriber).
- `../agentic-robotics-node` — NAPI bindings.
