# ruvbot / docs / adr

Architecture Decision Records (ADRs) capturing why the major design
choices were made. Numbered ADR-001 through ADR-015.

## Index
- 001 architecture-overview - DDD bounded contexts.
- 002 multi-tenancy-design - tenant/workspace isolation.
- 003 persistence-layer - Postgres + memory store strategy.
- 004 background-workers - bullmq + ioredis workers.
- 005 integration-layer - provider abstraction.
- 006 wasm-integration - embedding/index WASM modules.
- 007 learning-system - patterns, embeddings, training loop.
- 008 security-architecture - AIDefence + sandbox.
- 009 hybrid-search - BM25 + HNSW fusion.
- 010 multi-channel - Slack/Discord adapters.
- 011 swarm-coordination - Byzantine consensus & SwarmCoordinator.
- 012 llm-providers - Anthropic/Google/OpenRouter.
- 013 gcp-deployment - Cloud Run + Terraform.
- 014 aidefence-integration - prompt-injection guard.
- 015 chat-ui - public API and chat web UI.
