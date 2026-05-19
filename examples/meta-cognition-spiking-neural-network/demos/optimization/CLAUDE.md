# demos/optimization

Performance and adaptation demos for AgentDB's attention + SIMD stack.

## Files
- `package.json` - `@agentdb/simd-ops` v1.0.0. Scripts `test`/`benchmark` both run `simd-optimized-ops.js`.
- `adaptive-cognitive-system.js` - Self-optimizing system that learns which attention mechanism is best per task via performance tracking.
- `performance-benchmark.js` - Sweep benchmark across attention mechanisms and dimensions/batch sizes.
- `simd-optimized-ops.js` - SIMD-friendly vector op implementations targeting V8/SpiderMonkey auto-vectorization (claimed 5-54x speedup).

## Run
```
node performance-benchmark.js
node adaptive-cognitive-system.js
npm test   # simd-optimized-ops.js
```

## Related
- Parent: `../CLAUDE.md`.
- Guides: `../../docs/OPTIMIZATION-GUIDE.md`, `../../docs/SIMD-OPTIMIZATION-GUIDE.md`.
