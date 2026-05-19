# ruvbot / src / integration / providers

LLM provider adapters (ADR-012). Each provider implements the shared
`LLMProvider` interface; `RuvBot` selects the active one via
configuration.

## Files
- `AnthropicProvider.ts` - Wraps `@anthropic-ai/sdk` (Claude models).
- `OpenRouterProvider.ts` - Calls the OpenRouter HTTP API for access
  to many models behind one key.
- `GoogleAIProvider.ts` - Google Gemini provider via the AI Studio
  REST API.

Factory helpers (`createAnthropicProvider`, etc.) are re-exported by
`integration/index.ts` and consumed by `src/RuvBot.ts`.
