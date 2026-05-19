# ui/ruvocal/src/lib/components/chat/

Svelte components that compose the chat experience: the chat window, the message input, message rendering, markdown rendering, file/image handling, voice input, and tool/reasoning displays.

## Files

- `ChatWindow.svelte` — top-level chat layout (message list + input + model switch).
- `ChatIntroduction.svelte` — empty-state intro shown for new conversations.
- `ChatInput.svelte` — composer (text, file/url attachments, voice trigger, send).
- `ChatMessage.svelte` — single message renderer (user/assistant/tool roles).
- `MessageAvatar.svelte` — avatar for a message role.
- `MarkdownRenderer.svelte` (+ `MarkdownRenderer.svelte.test.ts`) — top-level markdown rendering.
- `MarkdownBlock.svelte` — renders a single parsed markdown block (uses `lib/utils/parseBlocks.ts`).
- `BlockWrapper.svelte` — wrapper around message blocks (highlight/copy/etc.).
- `Alternatives.svelte` — sibling-alternatives navigation (the conversation-tree branch switcher).
- `ModelSwitch.svelte` — inline model-picker for switching models mid-conversation.
- `FileDropzone.svelte`, `UploadedFile.svelte`, `ImageLightbox.svelte` — file/image attachment UI.
- `UrlFetchModal.svelte` — modal for fetching/attaching a URL (`POST /api/fetch-url`).
- `VoiceRecorder.svelte` — voice input recorder (works with `/api/transcribe`).
- `TaskGroup.svelte`, `ToolUpdate.svelte` — render tool-call / task progress updates streamed from the server.
- `OpenReasoningResults.svelte` — collapsible display of model "reasoning" output (see `lib/server/textGeneration/reasoning.ts`).

## Related

- Server streams these messages from `src/lib/server/textGeneration/`.
- Client stores backing these: `src/lib/stores/pendingMessage.ts`, `pendingChatInput.ts`, `backgroundGenerations*.ts`.
