# ruvbot / src / core

Core (domain) bounded context. Holds agent / session / skill entities,
bot configuration, and the high-level `ChatEnhancer` orchestrator.

## Files
- `BotConfig.ts` - `ConfigManager` and `BotConfig` (loads env, validates
  via zod).
- `BotState.ts` - `BotStateManager` tracking bot status & metrics.
- `ChatEnhancer.ts` - Combines skills, memory, hybrid search and LLM
  providers into one chat call (`createChatEnhancer`).
- `types.ts` - Domain types (`Result`, `BotEvent`, etc).
- `errors.ts` - `RuvBotError`, `ConfigurationError`, `InitializationError`.
- `index.ts` - Barrel; re-exports agent, session, skill submodules
  and `ChatEnhancer`.

## Subdirectories
- `agent/` - Agent aggregate root and behaviors.
- `entities/` - Plain entity types (Agent, Message, Session, ...).
- `session/` - Session lifecycle + state.
- `skill/` - Skill definitions and contracts.
