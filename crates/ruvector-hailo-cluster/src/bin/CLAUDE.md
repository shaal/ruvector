# ruvector-hailo-cluster/src/bin

CLI binaries. Each is registered as `[[bin]]` in `../../Cargo.toml`.

- `worker.rs` → `ruvector-hailo-worker` — gRPC server running on a Pi 5 + Hailo-8 (or CPU fallback). Feature-gated by `hailo` / `cpu-fallback`.
- `embed.rs` → `ruvector-hailo-embed` — stdin / `--text` → JSONL embedding lines.
- `fakeworker.rs` → `ruvector-hailo-fakeworker` — host-side mock worker; used by integration tests in `../../tests/`.
- `stats.rs` → `ruvector-hailo-stats` — fleet observability: TSV / JSON / Prometheus exposition.
- `bench.rs` → `ruvector-hailo-cluster-bench` — sustained-load benchmark harness.
- `mmwave-bridge.rs` → `ruvector-mmwave-bridge` — reads mmWave radar bytes via `ruvector-mmwave` and feeds the cluster.
- `ruview-csi-bridge.rs` — bridges Wi-Fi CSI samples from RuView nodes into the cluster.
- `ruvllm-bridge.rs` — bridge for the ruvLLM stack.
- `ruvllm-pi-worker.rs` — Pi-side LLM worker that uses the in-tree `ruvllm` engine (requires `ruvllm-engine` feature).

All bins share `--workers` / `--workers-file` / `--tailscale-tag` discovery flags and `--auto-fingerprint` / `--validate-fleet` safety flags.

See `../CLAUDE.md` and `../../deploy/` for systemd units.
