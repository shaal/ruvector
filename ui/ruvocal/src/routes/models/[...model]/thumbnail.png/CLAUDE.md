# ui/ruvocal/src/routes/models/[...model]/thumbnail.png/

Dynamic Open Graph thumbnail for a model — renders an SVG via satori then converts to PNG with `@resvg/resvg-js`.

## Files

- `+server.ts` — `GET` returns a PNG response (with appropriate cache headers).
- `ModelThumbnail.svelte` — Svelte component that lays out the OG image; rendered with `satori-html` against the bundled Inter fonts (`lib/server/fonts/`).
