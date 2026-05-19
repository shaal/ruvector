# demos/exploration

Autonomous-discovery demos combining SNN, AgentDB, attention, and SIMD to surface emergent capabilities.

## Files
- `package.json` - `@agentdb/autonomous-discovery` v1.0.0; depends on `@ruvector/core`, `@ruvector/attention`; peer-depends on `@agentdb/snn-simd`. Scripts: `discover`, `explore`, `test`.
- `cognitive-explorer.js` - Interactive cognitive explorer (neuromorphic semantic memory, attention-modulated STDP learning).
- `discoveries.js` - Headless run that logs emergent behaviors discovered by the hybrid architecture.
- `memory.bin` - Persisted explorer memory (regenerated on demand).

## Run
```
npm run discover     # discoveries.js
npm run explore      # cognitive-explorer.js
```

## Related
- Parent: `../CLAUDE.md`.
- Underlying SNN: `../snn/`.
- Reflective variant: `../self-discovery/`.
- Discovery write-up: `../../docs/DISCOVERIES.md`.
