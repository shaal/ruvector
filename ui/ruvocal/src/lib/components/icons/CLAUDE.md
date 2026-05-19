# ui/ruvocal/src/lib/components/icons/

Svelte components wrapping individual SVG icons (custom icons that aren't pulled from `unplugin-icons`/`@iconify`).

## Files

- `Logo.svelte`, `LogoHuggingFaceBorderless.svelte` — brand logos (the latter is the legacy HF chat-ui logo).
- `IconBurger.svelte`, `IconChevron.svelte`, `IconCheap.svelte`, `IconFast.svelte`, `IconNew.svelte`, `IconPro.svelte` — UI affordances and pricing/tier markers.
- `IconLoading.svelte` — spinner.
- `IconMoon.svelte`, `IconSun.svelte` — theme toggle.
- `IconShare.svelte`, `IconPaperclip.svelte` — action icons.
- `IconMCP.svelte`, `IconOmni.svelte`, `IconDazzled.svelte` — feature-specific glyphs (MCP, "Omni" multimodal, etc.).

## Conventions

- Each file is a self-contained Svelte component exporting an `<svg>` with optional `size`/`class` props.
- For standard icon sets prefer `@iconify-json/*` packages via `unplugin-icons` (see `../../../vite.config.ts`).
