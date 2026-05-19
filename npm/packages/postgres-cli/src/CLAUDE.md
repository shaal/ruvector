# postgres-cli/src

TypeScript source for `@ruvector/postgres-cli`. Compiled to `dist/` by `tsc` (ESM).

## Files

- `index.ts` — library entry. Re-exports `RuVectorClient` and command classes.
- `cli.ts` — `commander`-based CLI. Wires up subcommands: vector, attention, gnn, graph, learning, benchmark, sparse, hyperbolic, routing, quantization, install.
- `client.ts` — `RuVectorClient` PG wrapper with pooling, retries, batched ops, SQL injection protection; plus pool/retry config types and result interfaces.
- `commands/` — one module per command group.

Compiled `.js`/`.d.ts`/`.map` artifacts are present beside each `.ts` source.
