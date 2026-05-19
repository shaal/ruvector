Core TypeScript modules for the top-level benchmark suite. Invoked through `ts-node` from `../package.json`.

Files:
- `benchmark-runner.ts` - CLI entry point. Subcommands: `list`, `run <scenario>`, `group <name>`. Orchestrates load generation, metrics collection, and analysis.
- `benchmark-scenarios.ts` - declarative scenario + group registry (baselines, bursts, workload mixes, failover, real-world events).
- `load-generator.ts` - drives k6 (and direct HTTP) request patterns against a configured `BASE_URL`.
- `metrics-collector.ts` - captures latency/throughput/error metrics during a run.
- `results-analyzer.ts` - post-processes metrics into reports, computes performance scores (0-100), and writes summaries to `results/run-<ts>/`.

These modules are pure TS (no Rust). They assume a running ruvector cluster reachable via `BASE_URL` from `../.env`.
