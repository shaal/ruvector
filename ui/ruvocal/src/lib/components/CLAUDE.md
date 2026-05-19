# ui/ruvocal/src/lib/components/

General-purpose Svelte 5 components used across the chat UI (modals, navigation, banners, buttons, generic primitives). Domain-specific components are split out into subdirectories.

## Notable top-level components

- `Modal.svelte`, `Portal.svelte`, `HoverTooltip.svelte`, `Tooltip.svelte` — UI primitives.
- `NavMenu.svelte`, `MobileNav.svelte`, `ExpandNavigation.svelte`, `NavConversationItem.svelte` — navigation / sidebar.
- `WelcomeModal.svelte`, `AnnouncementBanner.svelte` — onboarding / messaging.
- `DeleteConversationModal.svelte`, `EditConversationModal.svelte`, `ShareConversationModal.svelte`, `SystemPromptModal.svelte`, `SubscribeModal.svelte`, `HtmlPreviewModal.svelte` — conversation-level modals.
- `CodeBlock.svelte`, `MarkdownBlock.svelte` (also see `chat/MarkdownBlock.svelte`), `CopyToClipBoardBtn.svelte` — content rendering helpers.
- `Pagination.svelte`, `PaginationArrow.svelte`, `InfiniteScroll.svelte` — list paging.
- `RetryBtn.svelte`, `StopGeneratingBtn.svelte`, `ScrollToBottomBtn.svelte`, `ScrollToPreviousBtn.svelte` — chat-action buttons.
- `Switch.svelte`, `Toast.svelte` — input/feedback primitives.
- `ModelCardMetadata.svelte` — renders metadata for a model card.
- `FoundationBackground.svelte`, `RuFloUniverse.svelte` — decorative background scenes (the latter is a three.js-driven "RuFlo" universe).
- `BackgroundGenerationPoller.svelte` — polls the server for in-flight background generations and updates the store.

## Subdirectories

- `chat/` — chat-window components (message list, input, file dropzone, etc.).
- `icons/` — SVG icon components.
- `mcp/` — MCP server management UI.
- `players/` — media players (audio).
- `voice/` — voice-input visualizations.
- `wasm/` — WASM/rvagent-specific UI panels.
