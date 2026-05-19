# prime-radiant/wasm/pkg

Generated/illustrative TypeScript artifacts for the Prime-Radiant WASM bindings (the `.wasm`/`.js` outputs from `wasm-pack` are produced into this dir on build; only the TS examples are checked in here).

## Files

- `prime_radiant_advanced_wasm.d.ts` - TypeScript declarations for the exported WASM API.
- `example.ts` - Example TypeScript usage showing how to instantiate engines and call exported methods.

## How to regenerate

```bash
cd /home/user/ruvector/examples/prime-radiant/wasm
wasm-pack build --target web --release
```

## Related

- Rust source: `../src/lib.rs`.
- Build instructions: `../CLAUDE.md`.
