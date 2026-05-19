Graph-database benchmark suite that exercises `crates/ruvector-graph` and compares it against Neo4j across synthetic social, knowledge, and temporal graphs.

Subdirectories:
- `src/` - TypeScript runners (data generator, scenarios, comparison runner, reporter, entry point).
- `docs/` - QUICKSTART and IMPLEMENTATION_SUMMARY for this graph suite.

Invocation (from `../`):
- `npm run graph:generate` - synthesize datasets to `benchmarks/data/graph/`.
- `npm run graph:bench` - run `cargo bench --bench graph_bench` in `crates/ruvector-graph`.
- `npm run graph:compare[:social|:knowledge|:temporal]` - run Neo4j comparisons.
- `npm run graph:report` - render reports.
- `npm run graph:all` - full pipeline.

Datasets target ~1M nodes / ~10M edges. Benchmarks measure node insertion, k-hop traversal, query latency, and throughput.
