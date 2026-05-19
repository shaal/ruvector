# src/cicd/

CI/CD synthetic data generator.

- `index.ts` — exports `CICDDataGenerator` plus types: `PipelineExecution`, `TestResults`, `DeploymentRecord`, `PerformanceMetrics`, `MonitoringAlert`, `PipelineStatus`.

Used to fabricate realistic pipeline executions, test runs, deployments, and monitoring alerts for training/evaluation.
