# ruvector-mincut/src/snn

Spiking neural-network cognitive-engine layer built on top of the min-cut graph: neurons + synapses form a dynamic graph whose cut signal drives a strange-loop cognitive controller. Benchmarked in `benches/snn_bench.rs`.

## Files

- `mod.rs` — façade.
- `neuron.rs` — `Neuron` (leaky integrate-and-fire variant).
- `synapse.rs` — `Synapse` connection model.
- `network.rs` — `Network` assembly + step loop.
- `attractor.rs` — attractor dynamics / state-space analysis.
- `causal.rs` — causal-graph inference.
- `morphogenetic.rs` — morphogenetic plasticity (grow/prune connections).
- `cognitive_engine.rs` — top-level cognitive controller driver.
- `optimizer.rs` — learning / parameter optimization.
- `strange_loop.rs` — strange-loop self-referential feedback.
