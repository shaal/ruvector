# ui/ruvocal/src/lib/server/__tests__/

Server-level integration tests for cross-cutting behavior that doesn't fit under a single module.

## Files

- `conversation-stop-generating.spec.ts` — verifies the conversation stop-generating flow (abort registry + endpoint integration). Pairs with `../abortRegistry.ts`, `../abortedGenerations.ts`, and `routes/conversation/[id]/stop-generating/+server.ts`.

Tests run via vitest with `scripts/setups/vitest-setup-server.ts`.
