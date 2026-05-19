# ui/ruvocal/src/lib/server/textGeneration/

End-to-end text-generation orchestration: takes a conversation + user message, runs preprocessing, calls the routed endpoint, optionally drives MCP tool calls, streams `MessageUpdate` events back to the client, and produces a saved assistant message.

## Files

- `index.ts` — public entrypoint; orchestrates the whole flow.
- `generate.ts` — wraps the model endpoint stream and yields normalized updates.
- `reasoning.ts` — handles model "reasoning" / thinking blocks (rendered by `chat/OpenReasoningResults.svelte`).
- `title.ts` — auto-generates conversation titles via the default endpoint (uses `lib/server/generateFromDefaultEndpoint.ts`).
- `types.ts` — shared types (params, stream events).

## Subdirectories

- `mcp/` — MCP-specific subflow: tool invocation, router resolution, file refs, WASM-tool tests.
- `utils/` — shared helpers (file prep, routing, tool prompt assembly).
