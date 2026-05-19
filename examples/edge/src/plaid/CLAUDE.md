# edge / src / plaid

PLAID (Privacy, Learning, AI, Distributed) subsystem. Implements local learning with Bulletproofs-backed zero-knowledge proofs of training behaviour. Ships native and WASM variants of both the prototype and the production-grade proof code.

## Important files
- `mod.rs` - module root.
- `zkproofs.rs` - native ZK-proof prototype on Bulletproofs.
- `zkproofs_prod.rs` - production-grade native ZK-proof implementation.
- `zk_wasm.rs` - WASM bindings around the prototype proofs.
- `zk_wasm_prod.rs` - WASM bindings around the production proofs.
- `wasm.rs` - non-ZK PLAID WASM surface (local-learner API exposed to JS).

## Build / run
- Native: included by default in `cargo build -p ruvector-edge`.
- WASM: `bash ../../scripts/build-wasm.sh`; outputs in `../../pkg/`.

## Related
- Browser demos: `../../pkg/plaid-demo.html`, `../../pkg/plaid-local-learner.ts`, `../../pkg/zk-demo.html`, `../../pkg/zk-financial-proofs.ts`.
- Performance docs: `../../docs/zk_*.md`, `../../docs/plaid-local-learning.md`. Bench: `../../benches/zkproof_bench.rs`.
