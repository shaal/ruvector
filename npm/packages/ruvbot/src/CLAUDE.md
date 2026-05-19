# ruvbot / src

All TypeScript source for the `ruvbot` package. Organized by DDD
bounded contexts plus shared utilities and the HTTP server entry.

## Top-level files
- `index.ts` - Public barrel: re-exports core, infrastructure,
  integration, learning, and the top-level `RuvBot` class.
- `RuvBot.ts` - Main framework class. Wires config, state, providers
  (Anthropic / OpenRouter / Google AI), sessions, agents, and events
  (built on `eventemitter3`).
- `server.ts` - HTTP entry point used in Cloud Run. Exposes health,
  chat, session, and agent REST endpoints; integrates the AIDefence
  guard and `ChatEnhancer`.
- `types.ts` - Shared branded types (`TenantId`, `WorkspaceId`,
  `UserId`, `AgentId`, `SessionId`, `TurnId`, `MemoryId`, etc).

## Subdirectories
- `core/` - Domain entities, sessions, skills, config, chat enhancer.
- `infrastructure/` - Persistence, messaging, worker abstractions.
- `integration/` (and legacy `integrations/`) - LLM providers, Slack,
  webhooks.
- `channels/` - Adapter registry for Slack/Discord/etc messaging.
- `learning/` - Embeddings, memory, patterns, search, training.
- `plugins/` - Plugin manager and plugin contract.
- `security/` - AIDefence guard integration.
- `skills/` - Skill execution + builtin skills.
- `swarm/` - Byzantine consensus + SwarmCoordinator.
- `templates/` - Conversation / agent templates.
- `api/` - REST API handlers and a public chat UI.
- `cli/` - Commander-based subcommands.
- `utils/` - Pino-based logger and helpers.

Compiled outputs live next to each `.ts` (CJS) and under `dist/esm/`
(ESM, via `tsconfig.esm.json`).
