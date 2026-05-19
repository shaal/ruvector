# raft / src

TypeScript source for `@ruvector/raft`.

## Files
- `index.ts` - Public barrel: re-exports node, state, log, and shared
  types as the package API.
- `node.ts` - `RaftNode` class. Handles election timeouts, heartbeats,
  RequestVote / AppendEntries RPCs, leader/follower/candidate state
  transitions, and applies committed entries to a user state machine
  via injected transport.
- `state.ts` - Persistent (`currentTerm`, `votedFor`, `log`) and
  volatile (`commitIndex`, `lastApplied`, leader-only nextIndex/
  matchIndex) state.
- `log.ts` - Append-only log with term metadata and conflict resolution.
- `types.ts` - Transport / RPC / event / config interfaces
  (`RaftTransport`, `NodeState`, `RaftConfig`, RPC payloads).

Each `.ts` ships with adjacent compiled `.js`, `.d.ts`, and source maps.
