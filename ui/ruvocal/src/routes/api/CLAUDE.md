# ui/ruvocal/src/routes/api/

Legacy HTTP API surface mounted at `/api/...`. Prefer the **v2** surface (`./v2/`) for new endpoints — it uses superjson and the reusable resolvers in `lib/server/api/`.

## Subdirectories

- `conversation/[id]/` — single-conversation operations (and `message/[messageId]/` for per-message).
- `conversations/` — list/create conversations.
- `fetch-url/` — fetches a URL on the server's behalf (CORS bypass for the URL-fetch modal); SSRF-checked via `lib/server/urlSafety.ts`.
- `mcp/` — MCP server health (`health/`) and CRUD for user-configured MCP servers (`servers/`).
- `models/` — model catalog.
- `transcribe/` — audio → text transcription (used by `lib/components/chat/VoiceRecorder.svelte`).
- `user/` — current-user info + `validate-token/`.
- `v2/` — newer API surface.
