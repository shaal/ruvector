# npm/core/native

Locally-built native binding artifacts. The loader in `../src/index.ts` checks `../native/<platform-arch>/` first before falling back to the per-platform npm packages under `../platforms/`. Useful during development when you've just built the NAPI-RS bindings on your own machine.

## Layout

- `<platform>-<arch>/index.cjs` - Thin CJS wrapper that requires `./ruvector.node` and remaps NAPI-RS's `VectorDb` (lowercase d) to the `VectorDB` shape expected by the TypeScript layer. Also wires `CollectionManager`, `version`, `hello`, `getMetrics`, `getHealth`, and `DistanceMetric` (from `JsDistanceMetric`).
- `<platform>-<arch>/ruvector.node` - Compiled NAPI-RS dynamic library for that target.

Only `linux-x64` is checked in here in this branch.

## Related

- `../platforms/` - Distribution form of the same artifacts, published as separate npm packages.
- `../src/index.ts` - The loader that consumes these files.
