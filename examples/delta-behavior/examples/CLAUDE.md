# delta-behavior / examples

Cargo `[[example]]` programs for the delta-behavior crate. Quick, runnable illustrations of the public API.

## Important files
- `demo.rs` - the minimal "hello delta-behavior" walkthrough: coherence measurement, transition gating, enforcement, and attractor guidance. Always buildable (no feature gate).

## Run
- `cargo run -p delta-behavior --example demo`.
- For other application examples (`self_limiting`, `swarm`, `containment`), see the `[[example]]` blocks in `../Cargo.toml`; each requires its matching feature.

## Related
- Full applications: `../applications/`. WASM equivalents: `../wasm/examples/`.
