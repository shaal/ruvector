# delta-behavior / applications

Eleven standalone application demos illustrating how delta-behavior constrains real systems. Each `.rs` file is buildable on its own once its feature flag is enabled in `../Cargo.toml`.

## Important files
- `01-self-limiting-reasoning.rs` - reasoning loop that refuses to diverge (feature `self-limiting-reasoning`).
- `02-computational-event-horizon.rs` - bounded "no-escape" compute frontier (feature `event-horizon`).
- `03-artificial-homeostasis.rs` - homeostatic control (feature `homeostasis`).
- `04-self-stabilizing-world-model.rs` - world model that snaps back to attractor basins (feature `world-model`).
- `05-coherence-bounded-creativity.rs` - generation gated on coherence (feature `coherence-creativity`).
- `06-anti-cascade-financial.rs` - cascade-resistant financial network (feature `anti-cascade`).
- `07-graceful-aging.rs` - graceful degradation over time (feature `graceful-aging`).
- `08-swarm-intelligence.rs` - delta-bounded swarm (feature `swarm-intelligence`).
- `09-graceful-shutdown.rs` - bounded-degradation shutdown (feature `graceful-shutdown`).
- `10-pre-agi-containment.rs` - capability containment example (feature `containment`).
- `11-extropic-substrate.rs` - extropic substrate experiment.
- `lib04_self_stabilizing_world_model.rmeta`, `lib08_swarm_intelligence.rmeta` - committed `rmeta` artifacts from prior builds (can be regenerated).

## Run
- `cargo run -p delta-behavior --example self_limiting --features self-limiting-reasoning` (see `../Cargo.toml` `[[example]]` table for canonical names).

## Related
- Library API consumed: `../src/lib.rs`. Theory: `../research/`. Domain model: `../ddd/`.
