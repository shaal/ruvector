Per-package publishing scripts. These are lower-level than `../deploy/deploy.sh` and can be run individually.

Files:
- `publish-all.sh` - triggers the GitHub Actions workflow that builds and publishes for all platforms.
- `publish-crates.sh` - publishes the Rust workspace crates to crates.io (handles dependency ordering).
- `publish-cli.sh` - one-liner that delegates CLI publishing.
- `publish-router-wasm.sh` - publishes the router WASM npm package.
- `check-and-publish-router-wasm.sh` - guarded variant that checks for version bumps first.

Requires `CRATES_API_KEY` and/or `NPM_TOKEN`. For full release flows prefer `../deploy/deploy.sh`.
