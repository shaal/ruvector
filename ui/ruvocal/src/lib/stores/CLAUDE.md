# ui/ruvocal/src/lib/stores/

Svelte stores (both classic `writable` and Svelte 5 `$state` rune-based `.svelte.ts` modules). Holds client-side reactive UI state.

## Files

- `settings.ts` — user settings (theme, model, system prompt, locale, etc.).
- `autopilotStore.svelte.ts` — state for autopilot chat mode (see ADR-037).
- `backgroundGenerations.svelte.ts` / `backgroundGenerations.ts` — tracks in-flight background generations (rune-based + plain variants).
- `pendingMessage.ts` — the message currently being streamed back from the server.
- `pendingChatInput.ts` — draft text/attachments in the composer.
- `mcpServers.ts` — user-configured MCP servers (paired with `lib/components/mcp/`).
- `wasmMcp.ts` — state for the in-browser WASM (`rvagent_wasm`) MCP tools.
- `isPro.ts` — whether the current user has a pro subscription.
- `isAborted.ts` — abort flag for the in-flight generation (also signaled to the server).
- `loading.ts` — global loading indicator.
- `errors.ts` — non-fatal error toasts/queue.
- `shareModal.ts` — open/close state for the share modal.
- `titleUpdate.ts` — drives the optimistic title-update UI after the server auto-titles a conversation.

## Conventions

- Files ending in `.svelte.ts` use Svelte 5 runes and may only be imported from `.svelte` or other `.svelte.ts` modules.
