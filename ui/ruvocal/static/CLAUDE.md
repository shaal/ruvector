# ui/ruvocal/static/

Static assets served as-is by SvelteKit at the site root.

## Files

- `robots.txt` — crawler directives.

## Subdirectories

- `chatui/` — branded assets for the default "chat-ui" / ruvocal branding (selected when `PUBLIC_APP_ASSETS=chatui`).
- `huggingchat/` — branded assets for the legacy HuggingChat branding (selected when `PUBLIC_APP_ASSETS=huggingchat`).
- `wasm/` — compiled WASM artifacts (`rvagent_wasm.js`, `rvagent_wasm_bg.wasm`) consumed by `src/lib/wasm/`.

## Conventions

- Branding is picked at build time via `PUBLIC_APP_ASSETS` (see `svelte.config.js`). Don't reference one branding folder's assets from another's code path.
