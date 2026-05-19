TypeScript/k6-based load-testing and end-to-end benchmark suite for the deployed ruvector cluster, published as the `@ruvector/benchmarks` npm workspace. Distinct from the in-repo Rust microbenches (`../benches/`) and the standalone `ruvector-bench` crate.

Top-level files:
- `package.json` - npm scripts: `test:quick`, `test:baseline`, `test:burst`, `test:standard`, `test:stress`, `test:reliability`, `test:full`, plus graph-specific `graph:*` runners.
- `setup.sh` - bootstraps k6, Node, ts-node, optional claude-flow; prompts for `BASE_URL` and writes `.env`.
- `Dockerfile` / `.dockerignore` - container image for running benchmarks.
- `visualization-dashboard.html` - browser dashboard for `results/*.json` (served via `npm run dashboard`).

Subdirectories:
- `src/` - core TS runners (benchmark-runner, scenarios, load-generator, metrics-collector, results-analyzer).
- `graph/` - graph-specific benchmark suite (data generation, Neo4j comparison, reporting).
- `vector-search/` - Python quantization benchmark (`benchmark_quantized_search.py`) + ANALYSIS.md write-up.
- `docs/` - QUICKSTART and LOAD_TEST_SCENARIOS reference.

Entry point is `ts-node src/benchmark-runner.ts {list|run <scenario>|group <name>}`. Results land in `results/run-<timestamp>/` (gitignored). Scenarios include `baseline_100m`, `baseline_500m`, `burst_10x/25x/50x`, `read_heavy`, `write_heavy`, `regional_failover`, `world_cup`, `black_friday`.

Requires k6 + Node 18+. Optional integration with `claude-flow` for hook-driven coordination.
