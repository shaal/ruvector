# agentic-synth/training

DSPy / OpenRouter training scripts and benchmark harnesses for `@ruvector/agentic-synth`.

## Key files

- `dspy-benchmarks.ts` — DSPy.ts benchmark suite.
- `dspy-learning-session.ts` — single-session learning runner.
- `dspy-multi-model-benchmark.ts` — multi-model benchmark.
- `dspy-real-integration.ts` — real DSPy integration test.
- `openrouter-learning-session.ts`, `openrouter-training-fixed.ts` — OpenRouter-driven training.
- `cli-runner.ts` — CLI to run any of the above.
- `run-benchmarks.ts`, `run-multi-model-benchmark.sh` — benchmark entry scripts.
- `example-usage.ts`, `example-output.json` — usage examples.
- `test-benchmark-import.cjs`, `test-dspy-integration.ts` — smoke tests.
- `BENCHMARKS_README.md`, `BENCHMARK_IMPLEMENTATION_SUMMARY.md`, `DSPY_INTEGRATION_README.md`, `IMPLEMENTATION_SUMMARY.md`, `INTEGRATION_COMPLETE.md`, `MULTI_MODEL_BENCHMARK_README.md`, `QUICK_START.md` — docs.

## Subdirectories

- `results/` — generated benchmark JSON results and training report.

Each `.ts` file has matching compiled `.js`, `.d.ts`, and source maps. Not bundled into the published npm package.
