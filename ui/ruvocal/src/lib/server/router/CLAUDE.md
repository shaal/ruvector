# ui/ruvocal/src/lib/server/router/

The **intelligent LLM router** — picks the best model/endpoint per request based on multimodal needs, tool availability, and policy. Documented in `docs/source/configuration/llm-router.md`.

## Files

- `arch.ts` — router architecture / classifier for routing decisions.
- `endpoint.ts` — wraps a routed decision into an endpoint call.
- `multimodal.ts` — multimodal capability detection (images/audio/video → vision-capable models).
- `policy.ts` — routing policy (cost/quality/latency tradeoffs, allow/block lists).
- `toolsRoute.ts` — tool-aware routing (forces tool-capable models when tools are bound).
- `types.ts` — router type definitions.

## Related

- Examples / presets: `lib/constants/routerExamples.ts`.
- Consumed by `../textGeneration/utils/routing.ts`.
