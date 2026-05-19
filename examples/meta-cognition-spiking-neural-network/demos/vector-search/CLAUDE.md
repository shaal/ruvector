# demos/vector-search

AgentDB / RuVector semantic-search demonstration over technical documentation. Showcases the claimed 150x speedup vs cloud-hosted alternatives.

## Files
- `semantic-search.js` - Builds a `VectorDB` from the `ruvector` package, indexes a small documentation corpus, runs semantic queries, and persists/reloads from `semantic-db.bin`.
- `semantic-db.bin` - On-disk index used by `semantic-search.js` (regenerated automatically).

## Run
```
node semantic-search.js
```

## Tech stack
- Node.js, `ruvector` (AgentDB) npm package.

## Related
- Parent: `../CLAUDE.md`.
- Other vector-search examples in `examples/` (e.g. `agentdb-vector-search` skill content).
