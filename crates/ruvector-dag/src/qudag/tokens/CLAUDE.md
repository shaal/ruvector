# ruvector-dag/src/qudag/tokens

rUv token integration for QuDAG — staking, governance voting, and reward distribution for pattern proposers/validators.

- `mod.rs` — re-exports; also contains the integration `#[cfg(test)] mod tests` (`test_staking_integration`, `test_rewards_calculation`).
- `staking.rs` — `StakingManager` (`new(min_stake, max_stake)`, `stake(node, amount, lock_days)`, `total_staked()`), `StakeInfo`, `StakingError`.
- `governance.rs` — `GovernanceSystem`, `GovernanceVote`, `Proposal`, `ProposalStatus`, `ProposalType`, `VoteChoice`, `GovernanceError`.
- `rewards.rs` — `RewardCalculator` (`pattern_validation_reward(stake, score)`), `RewardClaim`, `RewardSource`.

See `../CLAUDE.md`.
