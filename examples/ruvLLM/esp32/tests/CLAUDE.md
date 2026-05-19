# ruvLLM / esp32 / tests

Host-side simulation tests for `ruvllm-esp32`.

## Important files
- `simulation_tests.rs` - exercises the inference, quantization, federation, and ruvector subsystems on a host machine without an ESP32 attached.

## Run
- `cargo test` from `../` (standalone workspace).

## Related
- Host bench: `../benches/esp32_simulation.rs`.
