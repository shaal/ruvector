# src/analysis/

Static analysis modules used by the `ruvector` self-learning / decompiler pipeline.

- `index.ts` — barrel.
- `complexity.ts` — cyclomatic / cognitive complexity scoring.
- `patterns.ts` — pattern detection (design / anti-patterns).
- `security.ts` — security smell / vulnerability heuristics.

Each `.ts` ships alongside its compiled `.js` and `.d.ts`.
