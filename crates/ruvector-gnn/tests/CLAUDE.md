# ruvector-gnn/tests

Integration tests focused on loss convergence and verification of the GNN training pipeline.

## Files

- `loss_demo.rs` — end-to-end demo run exercising the optimizer + replay + EWC pipeline; useful as a worked example.
- `loss_verification.rs` — asserts loss trajectories converge under the expected schedules/optimizers.

Run: `cargo test -p ruvector-gnn`.
