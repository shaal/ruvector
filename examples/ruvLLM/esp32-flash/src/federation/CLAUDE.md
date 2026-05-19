# ruvLLM / esp32-flash / src / federation

Multi-chip federation layer for the flashable firmware (smaller surface than the library-only `../../../esp32/src/federation/`).

## Important files
- `mod.rs` - module root.
- `protocol.rs` - wire protocol between chips.
- `pipeline.rs` - pipeline-parallel scheduling.
- `speculative.rs` - speculative decoding for lower latency.

## Related
- Larger federation surface (sharding, tensor-parallel, FastGRNN router, scale modes): `../../../esp32/src/federation/`. Cluster scripts: `../../cluster-flash.*`, `cluster-monitor.sh`, `cluster.example.toml`.
