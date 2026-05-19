# src/workers/

Node `worker_threads` orchestration for `ruvector`.

- `index.ts` — barrel.
- `native-worker.ts` — runs the native `@ruvector/core` engine inside a worker for off-thread vector ops.
- `benchmark.ts` — worker-driven benchmarking harness.
- `types.ts` — worker message / job types.
