End-to-end deployment automation for publishing ruvector crates to crates.io and packages to npm.

Files:
- `deploy.sh` - main deployment driver (~23KB). Supports `--dry-run`. Requires `CRATES_API_KEY` and `NPM_TOKEN` env vars plus rust toolchain, Node 18+, `wasm-pack`, `jq`.
- `test-deploy.sh` - dry-run smoke test of the deploy pipeline.
- `DEPLOYMENT-QUICKSTART.md` - condensed 5-minute setup checklist.
- `DEPLOYMENT.md` - full deployment reference documentation.

Per-package publishers (one step lower) live in `../publish/`. Validation scripts in `../validate/` should be run before `deploy.sh`.
