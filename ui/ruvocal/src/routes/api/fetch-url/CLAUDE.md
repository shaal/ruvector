# ui/ruvocal/src/routes/api/fetch-url/

Server-side URL fetcher — lets the chat UI ingest a URL's content (used by `lib/components/chat/UrlFetchModal.svelte` and the `lib/workers/detailFetchWorker.ts`).

## Files

- `+server.ts` — `POST { url }`. Validates with `lib/server/urlSafety.ts` / `isURLLocal.ts` (SSRF protection), fetches via undici, sanitizes with isomorphic-dompurify, returns text/HTML/parsed content for attachment.
