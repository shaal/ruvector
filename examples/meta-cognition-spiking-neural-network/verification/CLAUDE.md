# meta-cognition-spiking-neural-network/verification

Verification scripts and report that exercise AgentDB and the demo suite to confirm they work end-to-end.

## Files
- `VERIFICATION-REPORT.md` - Human-readable report summarizing what was tested and the results.
- `functional-test.js` - Functional smoke tests over the demos.
- `verify-agentdb.js` - AgentDB-specific verification (load index, run query, check results).

## Run
```
node functional-test.js
node verify-agentdb.js
```

## Related
- Parent: `../CLAUDE.md`.
- Tests cover demos under `../demos/`.
