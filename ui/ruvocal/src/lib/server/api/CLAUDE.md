# ui/ruvocal/src/lib/server/api/

Helpers powering the **v2 API surface** (`src/routes/api/v2/...`). Provides shared types and reusable resolvers/middlewares for auth, conversation, and model lookup, plus a superjson-friendly response wrapper.

## Files

- `types.ts` — shared request/response types for the v2 API.

## Subdirectories

- `utils/` — composable resolvers:
  - `requireAuth.ts` — assert a valid session/token, returns the resolved user.
  - `resolveConversation.ts` — load a conversation by id with ownership checks.
  - `resolveModel.ts` — resolve a model id to a registered model config.
  - `superjsonResponse.ts` — wraps `Response` to serialize via superjson (matches `lib/APIClient.ts`).
- `__tests__/` — integration tests for the v2 endpoints (conversations, messages, user, reports).
