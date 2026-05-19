# ruvbot / src / learning / patterns

Pattern detection over the agent's interactions: clusters recurring
intents and behaviors so future chats can reuse them (ADR-007).

## Files
- `index.ts` - Barrel exposing the pattern store / matcher API
  consumed by `ChatEnhancer` and the training loop.
