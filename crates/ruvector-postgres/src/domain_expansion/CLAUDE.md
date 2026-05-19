# ruvector-postgres/src/domain_expansion

Cross-domain transfer learning for PostgreSQL — wraps `ruvector-domain-expansion` engines and exposes them as SQL functions, with a per-context DashMap cache.

## Files

- `mod.rs` — Global `DOMAIN_ENGINES: DashMap<String, Arc<RwLock<DomainExpansionEngine>>>` + `get_or_create_engine(context)`.
- `operators.rs` — pgrx SQL function wrappers.
