# prime-radiant/src/hott

Homotopy Type Theory module: types, paths, equivalences, transport, universes, and a small type checker.

## Files

- `mod.rs` - Module surface.
- `types.rs` - Core type representations.
- `term.rs` (~24KB) - Term/expression AST.
- `path.rs` - Path types and identity proofs.
- `equivalence.rs` - Type equivalences.
- `transport.rs` - Transport along paths.
- `universe.rs` - Type universes / hierarchy.
- `checker.rs` (~33KB) - Type checker / kernel.
- `coherence.rs` - Higher coherence verification.

## Related

- ADR: `../../docs/adr/ADR-003-homotopy-type-theory.md`.
- Tests: `../../tests/hott_tests.rs` (currently disabled in manifest).
