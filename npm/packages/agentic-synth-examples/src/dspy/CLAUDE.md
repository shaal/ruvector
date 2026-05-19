# src/dspy/

DSPy-style training session and multi-model benchmark suite. Built and published as a separate subpath entry: `@ruvector/agentic-synth-examples/dspy`.

- `index.ts` — barrel; re-exports the classes/types below.
- `training-session.ts` — `DSPyTrainingSession`, `ModelTrainingAgent`, `OptimizationEngine`, `TrainingPhase`, plus per-provider agents `ClaudeSonnetAgent`, `GPT4Agent`, `LlamaAgent`, `GeminiAgent`.
- `benchmark.ts` — `MultiModelBenchmark`, `BenchmarkCollector`, `ModelProvider`, and metric types (`QualityMetrics`, `PerformanceMetrics`, `IterationResult`, `BenchmarkMetrics`, `BenchmarkResult`, `ComparisonReport`).

Built via `npm run build:dspy` -> `tsup` into `dist/dspy/`.
