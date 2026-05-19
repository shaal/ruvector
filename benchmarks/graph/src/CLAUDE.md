TypeScript implementation of the graph benchmark suite. Entry point exports are re-exported via `index.ts` and called by the `npm run graph:*` scripts in `../../package.json`.

Files:
- `index.ts` - public exports (scenarios, generators, comparison runner, reporter) and a `runQuickBenchmark()` shortcut. Runs reporting if invoked directly.
- `graph-scenarios.ts` - scenario definitions and `datasets` catalog (social_network, knowledge_graph, temporal_events).
- `graph-data-generator.ts` - synthetic dataset generators (`generateSocialNetwork`, `generateKnowledgeGraph`, `generateTemporalGraph`, `generateAllDatasets`, `saveDataset`).
- `comparison-runner.ts` - executes ruvector-graph vs Neo4j comparison runs (`runComparison`, `runAllComparisons`).
- `results-report.ts` - renders markdown/HTML reports from raw benchmark JSON.

Outputs are written under `benchmarks/data/graph/` (datasets) and `results/` (benchmark output).
