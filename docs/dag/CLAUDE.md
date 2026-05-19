# docs/dag/

Design and implementation notes for the **DAG** subsystem - the directed-acyclic-graph execution model used for query plans, attention, and self-healing. Numbered docs read as an ordered series.

## Docs

- `00-INDEX.md` - section index.
- `01-ARCHITECTURE.md` - DAG architecture.
- `02-DAG-ATTENTION-MECHANISMS.md` - DAG-style attention mechanisms.
- `03-SONA-INTEGRATION.md` - SONA integration.
- `04-POSTGRES-INTEGRATION.md` - PostgreSQL integration.
- `05-QUERY-PLAN-DAG.md` - using DAGs for query plans.
- `06-MINCUT-OPTIMIZATION.md` - mincut-based DAG optimization.
- `07-SELF-HEALING.md` - self-healing DAG executor.
- `08-QUDAG-INTEGRATION.md` - QuDAG integration.
- `09-SQL-API.md` - SQL API surface.
- `10-TESTING-STRATEGY.md` - testing strategy.
- `11-AGENT-TASKS.md` - agent task decomposition.
- `12-MILESTONES.md` - implementation milestones.

## Related

- `../postgres/` - PostgreSQL integration docs.
- `../gnn/` - graph neural network integration.
- `../research/mincut/` - mincut algorithm research.
