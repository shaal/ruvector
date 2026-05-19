# rudag/bin

CLI entry point for the `@ruvector/rudag` package.

## Files

- `cli.js` — `rudag` CLI binary (referenced by `bin.rudag` in package.json). Lazy-loads `RuDag`, `DagOperator`, `AttentionMechanism` to keep startup fast. Validates file path arguments for security (prevents reading arbitrary files).

Used via `npx @ruvector/rudag <command>` after install.
