# agentic-synth/src/adapters

Adapters that bridge `@ruvector/agentic-synth` to optional peer packages.

## Files

- `ruvector.js` — adapter for the `ruvector` vector DB (insert generated samples as embeddings).
- `robotics.js` — adapter for `agentic-robotics`.
- `midstreamer.js` — adapter for `midstreamer` streaming framework.

Each peer is an optional `peerDependency` in package.json.
