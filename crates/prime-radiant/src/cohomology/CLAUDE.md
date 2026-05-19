# prime-radiant/src/cohomology

Sheaf cohomology for detecting global inconsistencies (obstructions) on the coherence graph.

Computes `H^0` (global sections) and `H^1` (obstructions to patching), using the coboundary operator `delta^0: C^0(G, F) -> C^1(G, F)` defined by `(delta^0 f)(e) = rho_t(f(t(e))) - rho_s(f(s(e)))`.

## Files

- `mod.rs` — math intro + module wiring.
- `sheaf.rs` — sheaf data structure over the substrate graph.
- `simplex.rs` — simplicial primitives for chain complexes.
- `cocycle.rs` — cocycle / coboundary representations.
- `cohomology_group.rs` — `H^n` group construction.
- `laplacian.rs` — sheaf Laplacian assembly (delta + delta^T composition).
- `obstruction.rs` — extracts obstruction classes from `H^1`.
- `diffusion.rs` — heat-equation diffusion on the sheaf for smoothing.
- `neural.rs` — neural restriction-map variant integration.

## Related

- Feeds back into `coherence/engine.rs` and `learned_rho/` for adaptive restriction maps.
