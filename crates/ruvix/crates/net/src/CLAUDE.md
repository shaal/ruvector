# ruvix-net/src

## Files

- `lib.rs` — crate root; re-exports `stack` and per-protocol modules.
- `stack.rs` — top-level `NetStack` orchestration.
- `device.rs` — `NetDevice` trait abstracting the underlying NIC.
- `ethernet.rs` — Ethernet II framing.
- `arp.rs` — ARP resolution / cache.
- `ipv4.rs` — IPv4 header + routing.
- `icmp.rs` — ICMP echo (ping) etc.
- `udp.rs` — UDP datagram transport.
- `error.rs` — net error enum.
