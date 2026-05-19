# ruvector-tiny-dancer-wasm/src

Single-file WASM facade for `ruvector-tiny-dancer-core`.

## Files

- `lib.rs` — `init()` panic hook + `RouterConfig` class with fluent setters (`model_path`, `confidence_threshold`, `max_uncertainty`, `enable_circuit_breaker`, `circuit_breaker_threshold`, `enable_quantization`). Wraps core `Router`, `Candidate`, `RoutingRequest`, `RoutingResponse`. Default config points to `./models/fastgrnn.safetensors`.
