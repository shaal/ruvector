# ruvix-integration/tests

Cross-subsystem integration tests for the RuVix kernel.

## Files

- `adr087_section17_acceptance.rs` — ADR-087 Section 17 acceptance criteria, validated end-to-end across nucleus + cap + region +
  queue + proof.
- `syscall_flows.rs` — multi-syscall flows (e.g. create region -> grant cap -> send queue msg -> prove + apply graph mutation).
