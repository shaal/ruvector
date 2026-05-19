# edge-net/tests

Cross-cutting tests for the `ruvector-edge-net` crate. Mixes Rust integration tests, Node/TS tests, manual scripts, and Docker-orchestrated multi-contributor tests.

## Important files
- Rust integration tests: `adversarial_scenarios_test.rs`, `economic_edge_cases_test.rs`, `learning_scenarios_test.rs`, `mcp_integration_test.rs`, `performance_benchmark.rs`, `rac_axioms_test.rs`.
- TS/JS tests: `credit-persistence.test.ts`, `qdag-persistence.test.ts`, `relay-security.test.ts`, `verify-credit-flow.js`, `manual-credit-test.cjs`.
- Docker harness: `docker-compose.test.yml`, `Dockerfile.contributor`, `Dockerfile.test-runner`.
- JS config: `package.json`, `tsconfig.json`, `jest.config.js`.

## Run
- Rust: `cargo test --features full` from `../`.
- TS/JS: `npm install && npx jest` (or per-file via Node).
- Docker harness: `docker compose -f tests/docker-compose.test.yml up --abort-on-container-exit`.

## Related
- Validation docs in `../docs/`.
