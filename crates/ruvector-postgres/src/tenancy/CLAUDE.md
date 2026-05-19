# ruvector-postgres/src/tenancy

Multi-Tenancy Module — first-class multi-tenancy with tenant-isolated vector search, per-tenant integrity monitoring, fair quotas, row-level security, and multiple isolation levels (shared, partition, dedicated).

## Files

- `mod.rs` — Module entry; exposes `ruvector_tenant_create(...)` family of SQL functions.
- `isolation.rs` — Isolation levels (Shared / Partition / Dedicated).
- `registry.rs` — Tenant registry.
- `operations.rs` — Tenant CRUD operations.
- `quotas.rs` — Per-tenant resource quotas.
- `rls.rs` — Row-Level Security integration.
- `validation.rs` — Tenant-name / config validation.
