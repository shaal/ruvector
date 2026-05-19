# ruvbot / src / swarm

Multi-agent swarm coordination (ADR-011). Provides a Byzantine fault-
tolerant consensus primitive plus a `SwarmCoordinator` to orchestrate
groups of agents.

## Files
- `ByzantineConsensus.ts` - PBFT-style consensus used to agree on
  collective decisions across agents.
- `SwarmCoordinator.ts` - Spawns / dispatches tasks across agents and
  aggregates their results.
- `index.ts` - Barrel re-exporting both.
