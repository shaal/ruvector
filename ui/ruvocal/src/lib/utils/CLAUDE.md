# ui/ruvocal/src/lib/utils/

Pure (mostly stateless) helpers for both client and server. Anything needing DB / env / secrets belongs in `lib/server/` instead.

## Notable files

- `marked.ts` (+ `.spec.ts`) — marked configuration (custom renderers, KaTeX integration).
- `parseBlocks.ts`, `parseIncompleteMarkdown.ts` — splits assistant output into renderable blocks; tolerant of mid-stream incomplete markdown.
- `messageUpdates.ts` (+ `.spec.ts`) — utilities for applying server `MessageUpdate` events to the local message state.
- `generationState.ts` (+ `.spec.ts`) — derives high-level generation state from a stream.
- `mcpValidation.ts` — validates user-entered MCP server configs (used by `components/mcp/AddServerForm.svelte`).
- `template.ts` (+ `.spec.ts`) — Handlebars template helpers for prompt templates.
- `PublicConfig.svelte.ts` — Svelte 5 rune-based holder for the public client config returned by `/api/v2/public-config`.
- `models.ts` — client-side model helpers (display name, capability flags).
- `auth.ts` — client-side auth helpers (login URL, token storage).
- `hf.ts` — HuggingFace-specific helpers.
- `favicon.ts` — dynamic favicon swap for activity indication.
- `mime.ts`, `searchTokens.ts`, `chunk.ts`, `debounce.ts`, `timeout.ts`, `sum.ts`, `randomUuid.ts`, `sha256.ts`, `hashConv.ts`, `getHref.ts`, `isUrl.ts`, `isDesktop.ts`, `isVirtualKeyboard.ts`, `cookiesAreEnabled.ts`, `haptics.ts`, `file2base64.ts`, `fetchJSON.ts`, `loadAttachmentsFromUrls.ts`, `mergeAsyncGenerators.ts`, `getReturnFromGenerator.ts`, `deepestChild.ts`, `formatUserCount.ts`, `parseStringToList.ts`, `stringifyError.ts` — small focused utilities.

## Subdirectories

- `tree/` — conversation-tree manipulation (add children/siblings, build subtree, convert legacy linear conversations, message-id checks). Pure functions with extensive `.spec.ts` coverage — touch carefully.
