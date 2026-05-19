# subpolynomial-time/src/fusion

Local "Vector-Graph Fusion" module exercised by the demo's scenario 7.
Combines vector relations with graph structure to monitor brittleness
and propose optimizer actions.

## Files

- `mod.rs` — public re-exports
  (`BrittlenessSignal`, `FusionConfig`, `FusionGraph`, `Optimizer`,
  `OptimizerAction`, `RelationType`, `StructuralMonitor`,
  `StructuralMonitorConfig`).
- `fusion_graph.rs` — `FusionGraph` data structure combining typed
  vector relations with the min-cut graph.
- `optimizer.rs` — `Optimizer` + `OptimizerAction` — recommends edge
  reinforcement / pruning under brittleness pressure.
- `structural_monitor.rs` — `StructuralMonitor` + config + emitted
  `BrittlenessSignal` events.

## Related

- `../main.rs` — scenario 7 driver
- `../../../../crates/ruvector-mincut/` — underlying min-cut crate
