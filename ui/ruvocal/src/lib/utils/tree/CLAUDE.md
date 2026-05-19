# ui/ruvocal/src/lib/utils/tree/

Pure helpers for manipulating the **message-tree** that backs each conversation (every message has a `parent`, alternative responses create siblings, so a conversation is a DAG rooted at a system message).

## Files

- `addChildren.ts` (+ `.spec.ts`) — append child message(s) under a parent.
- `addSibling.ts` (+ `.spec.ts`) — add a sibling alternative to an existing message.
- `buildSubtree.ts` (+ `.spec.ts`) — extract a subtree starting from a given message id.
- `convertLegacyConversation.ts` (+ `.spec.ts`) — converts pre-tree linear conversations into the tree format (used by `migrations/routines/`).
- `isMessageId.ts` (+ `.spec.ts`) — validates message-id shape.
- `treeHelpers.spec.ts` — cross-cutting tree helper tests.
- `tree.d.ts` — internal tree type declarations.

## Conventions

- All functions are **pure**: take the message array, return a new message array. Do not mutate inputs.
- Sibling/alternative-aware components (e.g. `lib/components/chat/Alternatives.svelte`) depend on these invariants — keep specs passing.
