# edge-net/src/credits

Credit ledger backed by a QDAG (Quantum-resistant DAG).

## Important files
- `mod.rs` — module entry; public ledger API.
- `qdag.rs` — QDAG data structure + persistence (see `../../docs/QDAG_ARCHITECTURE.md`).

## Related
- JS-side mirror: `../../pkg/{credits,ledger,qdag}.js`.
- UI: `../../dashboard/src/components/dashboard/CreditsPanel.tsx`.
