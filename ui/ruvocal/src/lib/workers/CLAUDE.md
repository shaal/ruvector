# ui/ruvocal/src/lib/workers/

Web Workers used to keep heavy work off the main UI thread.

## Files

- `autopilotWorker.ts` — drives the autopilot chat mode (see `lib/stores/autopilotStore.svelte.ts` and ADR-037).
- `detailFetchWorker.ts` — background fetching of detail/metadata for list items (e.g. conversation previews).
- `markdownWorker.ts` — markdown parsing/rendering off-thread; used by `lib/components/chat/MarkdownRenderer.svelte`.

## Conventions

- Workers are imported with Vite's worker import syntax (`?worker`). Keep their dependency surface small — Vite will bundle them separately.
