# edge-net/pkg

Pre-built `wasm-pack` output + Node.js orchestration layer that is publishable to npm as `@ruvector/edge-net`. Provides the `edge-net`, `ruvector-edge`, and `edge-net-join` CLI binaries and a rich JS-side runtime (DHT, signaling, ledger, agents, monitor, etc.).

## Important files
- `package.json` — npm manifest; declares CLI bin entries.
- `ruvector_edge_net.{js,d.ts}` + `ruvector_edge_net_bg.wasm{,.d.ts}` — web target WASM bundle.
- `node/` — same artifacts for the Node.js target.
- `cli.js`, `agents.js`, `real-agents.js`, `real-workflows.js`, `monitor.js`, `contribute-daemon.js`, `multi-contributor-test.js`, `contributor-flow-validation.cjs` — orchestration / contributor flow.
- `credits.js`, `ledger.js`, `qdag.js`, `sync.js` — credit ledger / QDAG.
- `dht.js`, `p2p.js`, `signaling.js`, `network.js`, `networks.js`, `genesis.js` — networking stack.
- `secure-access.js`, `firebase-setup.js` — auth/access integration.
- `join.html` + `join.js` — browser join UI.
- `models/` — JS-side model loader/registry/optimizer.
- `docs/migration-flow.md` — migration documentation.
- `Dockerfile`, `LICENSE`.

## Run
- `npm install -g @ruvector/edge-net` then `edge-net` / `edge-net-join`.
- Or from this dir: `node cli.js`.

## Related
- Rust source: `../src/`. WebUI: `../dashboard/`. Relay: `../relay/`.
