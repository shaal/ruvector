# ui/ruvocal/src/lib/server/textGeneration/utils/

Helpers reused across the text-generation orchestration.

## Files

- `prepareFiles.ts` — fetches uploaded file blobs, runs MIME/size checks, and shapes them into the format the endpoint expects (base64 / URLs / multipart parts).
- `routing.ts` — thin adapter over `../../router/` invoked from `../index.ts` to pick the model/endpoint.
- `toolPrompt.ts` — builds the tool-system-prompt fragment listing available MCP tools for models that need explicit prompting.
