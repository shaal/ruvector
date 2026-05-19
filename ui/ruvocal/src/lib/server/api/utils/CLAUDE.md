# ui/ruvocal/src/lib/server/api/utils/

Reusable resolvers/middlewares for v2 API `+server.ts` endpoints.

## Files

- `requireAuth.ts` — asserts the request has a valid session or API token, returns the resolved user/locals; throws a 401 otherwise.
- `resolveConversation.ts` — loads a conversation by `[id]` and asserts the caller can access it.
- `resolveModel.ts` — resolves a model id (namespace/model) to a config from `lib/server/models.ts`.
- `superjsonResponse.ts` — `Response` wrapper that JSON-serializes via `superjson` so the client (`lib/APIClient.ts`) can round-trip Dates, Maps, etc.

Compose these inside an endpoint, e.g.:

```ts
const user = await requireAuth(event);
const conv = await resolveConversation(event, user);
return superjsonResponse({ conv });
```
