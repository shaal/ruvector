# rvf/output

Captured `.rvf` / `.rvdna` artifacts produced by running the examples in `../examples/`. Useful as regression fixtures and for quick re-loading without rerunning examples.

## Notable files

- `claude_code_appliance.rvf` (~5.3 MB) - Largest sample store (Claude Code agent appliance).
- `progressive_index.rvf` (~2.6 MB), `quantization.rvf` (~1.5 MB) - Index/quantization demos.
- `legal_discovery.rvf` (~924 KB), `multimodal_fusion.rvf` (~823 KB), `embedding_cache.rvf` (~772 KB), `semantic_search.rvf` (~772 KB), `serverless.rvf` (~520 KB).
- Lineage / cow: `lineage_parent.rvf`, `lineage_child.rvf`, `lineage_snapshot.rvdna`.
- Reasoning chains: `reasoning_parent.rvf`, `reasoning_child.rvf`, `reasoning_grandchild.rvf`.
- Many smaller per-example outputs (agent_handoff, network_sync, posix_fileops, etc.).

## How to regenerate

```bash
cargo run -p rvf-examples --example <example_name>
```

## Related

- Producers: `../examples/`.
- Sample manifest: `../manifest.json`.
