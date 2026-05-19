# decompiler-dashboard

React + Vite single-page app for browsing, decompiling, and diffing
RuVector npm release artifacts across versions. Lets you pull a tarball,
split it into logical modules, run the decompiler/prettifier, and view
the source/RVF binary alongside metrics. Working demo (uses
`fetch`/`fetch -> blob` against the npm registry inside the browser).

## Important files

- `package.json` — Vite + React 18 + TypeScript SPA. Scripts:
  - `npm run dev` — local dev server
  - `npm run build` — `tsc -b && vite build`
  - `npm run preview` — preview built assets
  - `npm run prebuild:data` — runs `scripts/copy-research-data.mjs`
    (not present in tree — manual setup may be needed)
- `index.html` — Vite entry; loads JetBrains Mono + Inter and mounts
  `src/main.tsx`.
- `vite.config.ts`, `tsconfig.json`, `tailwind.config.js`,
  `postcss.config.js` — toolchain config (Vite 6, Tailwind 3, dark
  theme baseline).
- `src/App.tsx` — top-level routing (Explorer / Decompiler / RvfViewer)
  and per-version data fetch (`/data/{version}/source/metrics.json`).

## Tech stack

- React 18, react-router-dom v6, TypeScript 5
- Vite 6, Tailwind 3, PostCSS, autoprefixer
- `highlight.js` (syntax), `js-beautify` (prettifier), `jszip`
  (tarball unpack), `diff` (text diff)

## Related

- `src/components/`, `src/lib/`, `src/pages/`, `src/types/` — UI,
  business logic, route pages, type defs
- Sibling `../docs/` has standalone graph CLI docs but no shared deps
