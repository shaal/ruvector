# ruvbot / src / core / entities

Plain domain entities used across the core context.

## Files
- `Agent.ts` - Agent entity (id, persona, capabilities).
- `Message.ts` - Conversational `Message` (role, content, metadata).
- `Session.ts` - Conversation `Session` aggregating messages and
  participant metadata.

These entities are referenced by `agent/`, `session/`, and the
`ChatEnhancer` orchestrator.
