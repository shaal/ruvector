# ruvix-net

Minimal networking stack for the RuVix Cognition Kernel (ADR-087 Phase E). `no_std` with optional alloc.

## Layers

| Layer | Module | Purpose |
|---|---|---|
| Link | `ethernet` | Ethernet II frame handling |
| Network | `arp`, `ipv4`, `icmp` | Address resolution + IP routing |
| Transport | `udp` | Connectionless datagram transport |

## Files

- `Cargo.toml` — depends on `ruvix-types` (no-default-features). Dev: criterion, proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
