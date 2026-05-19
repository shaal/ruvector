# rvf/dashboard/dist

Pre-built production bundle for the RVF dashboard (output of `npm run build`).

## Files

- `index.html` - Built HTML entry.
- `rvf_solver_wasm.wasm` (~175 KB) - RVF solver WASM module.
- `assets/` - Hashed JS/CSS chunks (Three.js, D3, app bundle, styles).

## How to serve

```bash
cd /home/user/ruvector/examples/rvf/dashboard
npx vite preview
# or just serve the dist directory with any static server.
```

## Related

- Source: `../src/`.
- Manifest: `../package.json`.
