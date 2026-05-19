# ruvector-extensions/src/examples

Runnable in-source example programs for each major `ruvector-extensions` module.

## Files

- `embeddings-example.ts` — uses the embedding providers + `embedAndInsert` / `embedAndSearch` helpers.
- `persistence-example.ts` — JSON/binary/SQLite save+load and snapshot demos.
- `temporal-example.ts` — temporal tracking / version history walkthrough.
- `ui-example.ts` — boots the Express + WS UI server. Runnable via `npm run example:ui` (top-level package script invokes `tsx`).

Each `.ts` source has compiled `.js`, `.d.ts`, and `.map` siblings.
