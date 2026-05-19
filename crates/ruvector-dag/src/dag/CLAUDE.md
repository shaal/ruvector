# ruvector-dag/src/dag

Core DAG data structures and algorithms.

- `mod.rs` — module wiring; re-exports the four submodules.
- `query_dag.rs` — `QueryDag` (node + edge storage), construction APIs (`add_node`, `add_edge`), `DagError`.
- `operator_node.rs` — `OperatorNode`, `OperatorType` enum (`SeqScan`, `Filter`, `Join`, ...), constructors like `OperatorNode::seq_scan(id, table)`.
- `traversal.rs` — `BfsIterator`, `DfsIterator`, `TopologicalIterator`.
- `serialization.rs` — `DagSerializer` / `DagDeserializer` for persistence and over-the-wire transport.

See `../CLAUDE.md`.
