# dragnes / static

Static assets served verbatim by SvelteKit at the site root.

## Important files
- `manifest.json` - PWA web app manifest (installable on mobile, links to the icons below).
- `sw.js` - service worker; powers offline behaviour together with `$lib/dragnes/offline-queue.ts`.
- `dragnes-icon-192.svg`, `dragnes-icon-512.svg` - PWA icons at the manifest-required sizes.

## Related
- Offline queue logic: `../src/lib/dragnes/offline-queue.ts`. Page shell: `../src/app.html`.
