# prime-radiant/src/learned_rho

Learned restriction maps (rho) — replaces hand-coded rho with GNN-trained maps from `ruvector-gnn`. Each edge gets its own rho_u, rho_v producing the residual `r_e`.

## Files

- `mod.rs` — module entry.
- `config.rs` — training hyperparameters (layers, dim, lr).
- `map.rs` — `LearnedRho` runtime: applies the trained map at inference.
- `training.rs` — training loop with EWC/replay-buffer for continual learning.
- `error.rs` — module errors.

## Related

- `crates/ruvector-gnn` — underlying GNN layers (RuvectorLayer, ElasticWeightConsolidation, ReplayBuffer).
