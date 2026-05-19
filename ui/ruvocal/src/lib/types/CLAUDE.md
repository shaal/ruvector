# ui/ruvocal/src/lib/types/

Shared TypeScript domain types used by both client and server. Most align with MongoDB collections; reuse them rather than redefining shapes.

## Files

- `Conversation.ts`, `SharedConversation.ts`, `ConvSidebar.ts`, `ConversationStats.ts` — conversation entity, share variant, sidebar summary, computed stats.
- `Message.ts`, `MessageUpdate.ts`, `MessageEvent.ts` — message + the streaming update/event types the server emits.
- `Model.ts` — model config (id, params, capabilities, multimodal flags).
- `User.ts`, `Session.ts`, `Settings.ts` — user/session/settings shapes.
- `Assistant.ts`, `AssistantStats.ts` — custom assistant entity (legacy/HF chat-ui concept).
- `Tool.ts` — MCP tool descriptor.
- `Report.ts`, `Review.ts` — moderation report and review states.
- `Template.ts` — prompt template descriptor.
- `TokenCache.ts`, `AbortedGeneration.ts`, `MigrationResult.ts`, `Timestamps.ts`, `ConfigKey.ts`, `UrlDependency.ts`, `Semaphore.ts` — supporting types.

## Conventions

- Types here are pure (no runtime exports). Add new collection-backed types here.
