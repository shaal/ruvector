# ruvllm/src/training

Training data generation and fine-tuning for RuvLTRA models. Includes Claude
Flow task datasets, MCP tool training (140+ tools), GRPO RL, and contrastive
embedding training.

## Files
- `mod.rs` - public API + usage docs.
- `claude_dataset.rs` - task-routing dataset generation from Claude Flow.
- `tool_dataset.rs` - MCP tool calling dataset generation (140+ tools).
- `mcp_tools.rs` - `McpToolTrainer` (GRPO-based fine-tuning for tool use).
- `grpo.rs` - GRPO (Group Relative Policy Optimization) implementation.
- `contrastive.rs` - contrastive training for embedding heads.
- `real_trainer.rs` - production training loop driver.
- `claude_dataset.rs` / `tool_dataset.rs` (above) - dataset configs and
  emitters.
- `tests.rs` - local unit tests for this module.
