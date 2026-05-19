# neural-trader/exotic

Experimental ML/finance approaches: GNNs, attention, RL, quantum-inspired optimization, hyperbolic geometry, multi-agent swarms, atomic arbitrage.

## Files
- `atomic-arbitrage.js` - Cross-exchange flash-loan arbitrage with MEV protection.
- `attention-regime-detection.js` - Transformer self-attention for market regime identification across time scales.
- `benchmark.js` - Performance benchmark suite measuring execution time, memory, and throughput across the exotic examples.
- `gnn-correlation-network.js` - GNN over a correlation graph for feature extraction.
- `hyperbolic-embeddings.js` - Poincaré disk embeddings for hierarchical market structure.
- `multi-agent-swarm.js` - Specialized agents (momentum, mean-reversion, sentiment, arbitrage) reaching consensus.
- `quantum-portfolio-optimization.js` - QAOA-style portfolio optimization with simulated quantum annealing.
- `reinforcement-learning-agent.js` - Deep Q-Network with vector-similarity-driven experience replay.

## Run
```
npm run exotic:swarm
npm run exotic:gnn
npm run exotic:attention
npm run exotic:rl
npm run exotic:quantum
npm run exotic:hyperbolic
npm run exotic:arbitrage
node exotic/benchmark.js
```

## Related
- Parent: `../CLAUDE.md`.
- Used together with `../neural/`, `../portfolio/`.
