# ruvLLM / tests

Integration tests for the ruvLLM crate.

## Important files
- `integration.rs` - end-to-end tests covering inference, router, memory, orchestrator wiring.
- `sona_integration.rs` - integration tests for the SONA continual-learning subsystem (`../src/sona/`).

## Run
- `cargo test -p ruvllm` (add `--features ...` to exercise feature-gated paths).

## Related
- Benches: `../benches/`. Spec: `../docs/SONA/`.
