# ui/ruvocal/src/lib/actions/

Svelte `use:` actions — small reusable DOM behaviors attached to elements via `use:actionName`.

## Files

- `clickOutside.ts` — fires a custom event when a click occurs outside the element (used by modals/menus to close on outside click).
- `snapScrollToBottom.ts` — keeps a scrollable container pinned to the bottom while new content is appended (used in the chat message list).
