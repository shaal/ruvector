# ruvbot / src / api

HTTP/REST API handlers for ruvbot's chat, session, and admin endpoints,
plus the static chat UI served at runtime.

## Files
- `index.ts` - Barrel exporting the API route handlers wired up by
  `server.ts`.
- `public/` - Static assets shipped with the API (chat web UI).
