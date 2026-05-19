# ui/ruvocal/src/lib/server/textGeneration/mcp/

The MCP-tool subflow inside the text-generation pipeline: resolves which MCP server/tool a model wants to call, executes it, integrates file references, and feeds the result back into the stream.

## Files

- `runMcpFlow.ts` — orchestrates a full MCP tool-call round inside a generation.
- `toolInvocation.ts` — performs a single tool invocation against the MCP client pool.
- `routerResolution.ts` — resolves tool/router decisions when the LLM router selected an MCP-capable model.
- `fileRefs.ts` — translates MCP tool outputs that reference files into proper attachments / message updates.
- `wasmTools.test.ts` — tests for the in-browser WASM-tool path (`rvagent_wasm`) bridged through MCP.

## Related

- MCP clients: `../../mcp/`.
- Tool UI: `lib/components/chat/ToolUpdate.svelte`, `TaskGroup.svelte`.
