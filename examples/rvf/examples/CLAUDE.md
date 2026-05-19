# rvf/examples

~70 Cargo `[[example]]` binaries covering the full RVF stack and end-to-end agentic / scientific / security / dashboard demos.

## Highlights

- Core store: `basic_store.rs`, `progressive_index.rs`, `quantization.rs`, `wire_format.rs`, `crypto_signing.rs`, `filtered_search.rs`, `snapshot_freeze.rs`, `cow_branching.rs`, `dedup_detector.rs`, `sparse_matrix_store.rs`, `membership_filter.rs`.
- Agentic memory: `agent_memory.rs`, `agent_handoff.rs`, `swarm_knowledge.rs`, `reasoning_trace.rs`, `tool_cache.rs`, `experience_replay.rs`, `embedding_cache.rs`, `ruvbot.rs`, `claude_code_appliance.rs`.
- Runtimes: `browser_wasm.rs`, `edge_iot.rs`, `serverless_function.rs`, `ebpf_accelerator.rs`, `linux_microkernel.rs`, `self_booting.rs`, `live_boot_proof.rs`, `posix_fileops.rs`, `network_interfaces.rs`, `network_sync.rs`, `mcp_in_rvf.rs`, `openfang.rs`.
- Security: `security_hardened.rs` (~59KB), `sealed_engine.rs`, `tee_attestation.rs`, `zero_knowledge.rs`, `access_control.rs`.
- ML/RAG: `ruvllm_inference.rs`, `semantic_search.rs`, `recommendation.rs`, `rag_pipeline.rs`, `multimodal_fusion.rs`, `hyperbolic_taxonomy.rs`, `brain_training_integration.rs`.
- Scientific: `planet_detection.rs`, `microlensing_detection.rs`, `real_microlensing.rs`, `exomoon_graphcut.rs`, `life_candidate.rs`, `habitability_bias.rs`, `causal_atlas.rs`, `causal_atlas_dashboard.rs`, `causal_atlas_sealed.rs`, `climate_tipping.rs`, `climate_graphcut.rs`, `seismic_risk.rs`, `cyber_threat_graphcut.rs`, `financial_fraud_graphcut.rs`, `financial_signals.rs`, `genomic_graphcut.rs`, `genomic_pipeline.rs`, `medical_graphcut.rs`, `medical_imaging.rs`, `legal_discovery.rs`, `supply_chain_graphcut.rs`, `qaoa_graphcut.rs`, `solver_benchmark.rs`, `solver_witness.rs`, `real_data_discovery.rs`, `postgres_bridge.rs`, `generate_all.rs`.
- Subdir: `assets/` - large image/markdown asset(s).

## How to run

```bash
cargo run -p rvf-examples --example basic_store
cargo run -p rvf-examples --example causal_atlas_dashboard --release
cargo run -p rvf-examples --example security_hardened --release
```

## Related

- Crate root: `examples/rvf/`.
- Run outputs: `examples/rvf/output/`.
- Dashboard consumer: `examples/rvf/dashboard/`.
