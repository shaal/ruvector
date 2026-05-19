# ui/ruvocal/src/lib/server/endpoints/openai/

OpenAI-compatible endpoint adapter. Works against any OpenAI-API-shaped server (HuggingFace router, Ollama, llama.cpp, OpenRouter — see `docs/source/index.md`).

## Files

- `endpointOai.ts` — main adapter; builds the `openai` SDK client from env (`OPENAI_BASE_URL`, `OPENAI_API_KEY`), invokes chat / completion / embedding APIs, and returns a streaming `TextGenerationStream`.
- `openAIChatToTextGenerationStream.ts` — converts an OpenAI chat-completion stream into the app's internal `TextGenerationStream` (handles deltas, tool calls, reasoning blocks).
- `openAICompletionToTextGenerationStream.ts` — same conversion for the legacy `/completions` endpoint (text-only).

## Related

- Consumed by `../endpoints.ts`.
- Stream consumers: `lib/server/textGeneration/generate.ts`.
