# ui/ruvocal/src/routes/api/v2/conversations/

v2 conversation collection endpoints.

## Files

- `+server.ts` — `GET` list / `POST` create. Uses `lib/server/api/utils/requireAuth.ts` + superjson response.

## Subdirectories

- `[id]/` — single-conversation routes (+ nested `message/[messageId]/`).
- `import-share/` — import a shared conversation into the caller's account.
