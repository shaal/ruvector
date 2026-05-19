Smoke and integration test runners that exercise published packages and CLI surfaces.

Files:
- `test-wasm.mjs` - tests `ruvector-math-wasm` and `ruvector-attention-wasm` in Node.js.
- `test-docker-package.sh` - validates the docker-packaged build.
- `test-graph-cli.sh` - exercises the graph CLI commands.
- `test-all-graph-commands.sh` - extended graph CLI coverage.

Heavier integration tests live in `../../tests/` (Rust + TS); these scripts are mostly post-publish sanity checks.
