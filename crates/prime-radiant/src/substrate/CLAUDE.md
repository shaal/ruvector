# prime-radiant/src/substrate

The graph substrate the coherence engine operates over: nodes hold local state, edges carry restriction maps `rho_u`, `rho_v` whose disagreement is the residual `r_e`.

## Files

- `mod.rs` — module entry.
- `node.rs` — node value object (id, local section / state).
- `edge.rs` — edge value object (endpoints, weight, restriction maps).
- `graph.rs` — `Substrate` graph container.
- `restriction.rs` — `RestrictionMap` trait + hand-coded variants.
- `repository.rs` — substrate persistence interface.

## Related

- Learned restriction maps live in `learned_rho/`.
- Sheaf cohomology over this substrate is in `cohomology/`.
