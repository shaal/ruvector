# examples/meta-cognition-spiking-neural-network

Comprehensive AgentDB / RuVector demonstration suite combining spiking neural networks (SNN), attention mechanisms, SIMD-optimized vector ops, and meta-cognitive self-discovery. Originally from the "vibecast" weekly live-coding series.

## Top-level files
- `package.json` - npm package `vibecast` v1.0.0. Dependencies: `agentdb ^2.0.0-alpha.2.11`, `better-sqlite3 ^12.5.0`.
- `LICENSE` - MIT.

## Subdirectories
- `demos/` - Runnable demonstrations: attention mechanisms, exploration, optimization, self-discovery, SNN (SIMD N-API addon), vector-search. Top-level `demos/run-all.js` orchestrates them in sequence.
- `docs/` - Markdown guides for each subsystem (SNN, hyperbolic attention, SIMD, optimization, AgentDB exploration, discoveries).
- `verification/` - Functional verification scripts and a verification report.

## Run
```
node demos/run-all.js
# or individual demos
node demos/vector-search/semantic-search.js
```

## Tech stack
- Node.js >=16, AgentDB v2 alpha, SQLite (`better-sqlite3`).
- Native N-API C++ SNN with SIMD (under `demos/snn/native/`).

## Related
- AgentDB tutorials in this monorepo's `examples/` (vector-search, attention, neural-trader).
