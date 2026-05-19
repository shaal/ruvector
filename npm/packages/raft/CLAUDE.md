# @ruvector/raft

Pure-TypeScript implementation of the Raft consensus algorithm for
distributed systems: leader election, log replication, and fault
tolerance. Mirrors functionality available in the Rust
`ruvector-raft` crate, but for Node-only consumers.

## Important files
- `package.json` - npm metadata (`@ruvector/raft` v0.1.0). Single entry
  via `dist/index.js`. Depends on `eventemitter3`.
- `src/index.ts` - Public API barrel re-exporting `RaftNode`,
  `RaftTransport`, `NodeState`, log/state types.
- `src/node.ts` - `RaftNode` implementation (election timer, RPCs,
  state-machine application).
- `src/state.ts` - Persistent and volatile node state.
- `src/log.ts` - Replicated log abstractions.
- `src/types.ts` - Shared types/interfaces (transport, RPCs, events).
- `tsconfig.json` - Compiles `src/` to `dist/` with declarations.

## Exports / entry
- `main` -> `dist/index.js`, `types` -> `dist/index.d.ts`. Conditional
  `exports.import`/`exports.require` both serve the same compiled file.

## Scripts
- `build` - `tsc`.
- `test` - `node --test test/*.test.js` (no test/ dir checked in yet).
- `typecheck`, `clean`, `prepublishOnly` (-> `build`).

## Related
- Rust crate: `../../../crates/ruvector-raft`.
