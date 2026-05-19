# edge / docs

Design + performance documentation for the edge crate, especially the PLAID local-learning and ZK-proof subsystems.

## Important files
- `plaid-local-learning.md` - architecture of the PLAID (Privacy, Learning, AI, Distributed) local-learner.
- `zk_optimization_example.md` - worked example of optimizing a ZK proof.
- `zk_optimization_quickref.md` - cheat-sheet for the same.
- `zk_performance_analysis.md` - detailed performance analysis of the Bulletproofs pipeline.
- `zk_performance_summary.md` - high-level performance summary.

## Related
- ZK implementation: `../src/plaid/zkproofs.rs`, `../src/plaid/zkproofs_prod.rs`. Bench: `../benches/zkproof_bench.rs`. Demos: `../pkg/zk-demo.html`, `../pkg/zk-financial-proofs.ts`.
