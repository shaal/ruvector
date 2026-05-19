# exo-core/src/backends

Optional / feature-gated backend implementations declared inside
exo-core itself. The fully-fleshed-out classical backend lives in its
own crate (`exo-backend-classical`); these two are stubs / research
sketches kept inline for experimentation.

## Files

- `mod.rs` — module gate / backend selector.
- `neuromorphic.rs` — neuromorphic compute backend sketch (event-driven
  spiking semantics). See `../../../research/01-neuromorphic-spiking/`
  for the full prototype.
- `quantum_stub.rs` — placeholder API for a future quantum backend.

## Related

- `../traits.rs` — backend trait contracts these implement.
- `../../../exo-backend-classical/` — production-shaped backend.
- `../../../research/02-quantum-superposition/` — quantum cognition R&D.
