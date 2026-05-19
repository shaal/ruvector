# ruvbot / src / security

Security guards (ADR-008, ADR-014). Wraps the external `aidefence`
package to block prompt injection, jailbreaks, and PII leakage in
incoming/outgoing messages.

## Files
- `AIDefenceGuard.ts` - `createAIDefenceGuard(config)` factory and the
  `AIDefenceGuard` class used by `server.ts` and `ChatEnhancer`.
- `index.ts` - Barrel re-exporting the guard API.
