# src/

TypeScript source for `@ruvector/cli`. Compiled to `../dist/` via `tsc`.

- `cli.ts` — main CLI; defines commands via `commander`, handles self-learning hooks, and is published as the `ruvector` bin.
- `storage.ts` — storage abstractions (filesystem + optional Postgres via `pg`).
- `*.js`, `*.d.ts`, `*.map` — checked-in build artifacts mirroring the `.ts` files.
