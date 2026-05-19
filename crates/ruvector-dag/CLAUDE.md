# ruvector-dag

Directed Acyclic Graph (DAG) structures and algorithms for query-plan optimisation with neural learning capabilities. Combines a core DAG data structure with seven attention mechanisms, sub-polynomial mincut bottleneck detection, a self-optimising neural architecture (SONA) with MicroLoRA, autonomous self-healing, and a QuDAG (quantum-resistant distributed) consensus / token layer.

## Features

- `default = ["full"]`
- `full = ["tokio", "dashmap", "crossbeam", "parking_lot"]` — non-WASM build.
- `wasm = ["getrandom/js"]` — minimal feature set (core DAG + attention only).
- `production-crypto = ["pqcrypto-dilithium", "pqcrypto-kyber"]` — enables real ML-DSA / ML-KEM. Without it the QuDAG crypto module uses HMAC/HKDF placeholders.

## Layout

- `Cargo.toml` — see features above; deps include `ruvector-core`, `ndarray`, `zeroize`, `sha2`, optional `tokio`.
- `src/lib.rs` — crate root; conditionally compiles `sona`, `healing`, `qudag` only with `full`.
- `src/dag/` — core DAG data structures (`QueryDag`, `OperatorNode`, traversals, serialization).
- `src/attention/` — seven DAG-topology-aware attention mechanisms.
- `src/mincut/` — sub-polynomial mincut + bottleneck engine.
- `src/healing/` — anomaly detection, drift, repair strategies.
- `src/sona/` — Self-Optimising Neural Architecture (MicroLoRA, EWC++, reasoning bank).
- `src/qudag/` — quantum-resistant consensus, crypto, governance tokens.
- `benches/dag_benchmarks.rs` — criterion harness.
- `examples/` — usage demos (`basic_usage`, `attention_demo`, `self_healing`, `learning_workflow`, `synthetic_haptic`, `attention_selection`) plus `exotic/` advanced scenarios.
- `tests/` — `mincut_tests.rs` plus a fully-structured `integration/` and `fixtures/` tree.
- `.swarm-status.json` — bookkeeping for the agent-swarm workflow that wrote this crate.

## Public API (re-exported from `lib.rs`)

`QueryDag`, `OperatorNode`, `OperatorType`, `DagError`, `DagSerializer`, `DagDeserializer`, traversal iterators (`BfsIterator`, `DfsIterator`, `TopologicalIterator`); mincut types (`Bottleneck`, `BottleneckAnalysis`, `DagMinCutEngine`, `FlowEdge`, `LocalKCut`, `MinCutConfig`); attention configs/traits/scores; healing orchestrator; SONA engine; QuDAG client.

## Related

- `crates/ruvector-core` — base type system.
- `crates/ruvector-mincut` and `crates/cognitum-gate-kernel` — gate-fabric counterparts.
- `crates/ruvector-learning-wasm` — companion WASM crate sharing the MicroLoRA design.
- `crates/ruvector-math` — geometric primitives used in hierarchical-Lorentz attention and SONA.
