# rvlite / bin

Executable entry point for the `rvlite` CLI.

## Files
- `cli.js` - Shebang script wired up by `package.json`'s `bin` field
  as `rvlite`. Uses commander + chalk + ora to expose subcommands for
  managing the lightweight vector database (insert, search, SQL,
  Cypher, SPARQL). Backed at runtime by the compiled SDK in `dist/`
  and the RVF helper in `src/cli-rvf.ts`.
