# ruvector-robotics/src/cognitive

Layered cognitive architecture for autonomous robots: perceive → think → act → learn.

## Files

- `mod.rs` — module decls + cognitive-stack docs.
- `behavior_tree.rs` — composable BT nodes (sequence/selector/parallel/leaf actions).
- `cognitive_core.rs` — main perceive-think-act-learn loop tying the layers together.
- `decision_engine.rs` — multi-criteria utility-based action selection.
- `memory_system.rs` — working / episodic / semantic memory tiers.
- `skill_learning.rs` — motor-skill acquisition and refinement.
- `swarm_intelligence.rs` — multi-robot coordination primitives.
- `world_model.rs` — internal environment representation, used by the decision engine.
