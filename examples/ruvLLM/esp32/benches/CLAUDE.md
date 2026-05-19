# ruvLLM / esp32 / benches

Host-side benchmark that simulates ESP32 constraints (memory, FLOPS, INT8/INT4 paths).

## Important files
- `esp32_simulation.rs` - the single bench file.

## Run
- `cargo bench` from `../` (this is a standalone workspace).

## Related
- Production code being measured: `../src/`. Companion tests: `../tests/simulation_tests.rs`.
