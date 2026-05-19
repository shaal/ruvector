# rvlite/src/storage

Persistence + concurrency layer. Backs all three query engines (Cypher,
SPARQL, SQL) and exposes serializable state types used by `save()` /
`load()` in `../lib.rs`.

## Files
- `mod.rs` - public re-exports (`GraphState`, `RvLiteState`,
  `TripleStoreState`, `VectorState`).
- `state.rs` - top-level engine state aggregating vector / graph / triple
  components.
- `indexeddb.rs` - browser IndexedDB persistence (via `web-sys`).
- `id_map.rs` - bidirectional ID interning for nodes / triples / vectors.
- `epoch.rs` - epoch tracking for change detection.
- `writer_lease.rs` - single-writer lease to coordinate concurrent
  modifications.
