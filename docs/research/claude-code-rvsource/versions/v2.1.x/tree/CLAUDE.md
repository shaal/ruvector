# docs/research/claude-code-rvsource/versions/v2.1.x/tree/

Per-feature clusters of extracted JS modules from claude-code v2.1.x. Each subdirectory groups files around a representative symbol/identifier discovered during extraction. These are **machine-generated artifacts**, not curated documentation.

## Clusters

- `asyncgenerator/` - async generator + assorted utility shims.
- `bedrockclient/` - AWS Bedrock client modules.
- `is_wsl_test/` - WSL detection + utf8.
- `managedidentitycredential/` - Azure ManagedIdentityCredential.
- `mutatevalue/` - mutate-value helpers.
- `object_undefined/` - object/string/symbol normalization.
- `react_memo_cache_sentinel/` - large React memo cluster (has nested `additionaldirectoriesforclaudemd/` and `react_memo_cache_sentinel/` sub-buckets).
- `remote-settings_json/` - remote settings handling.
- `select-pane/` - terminal pane selection (iterm2 / backend).
- `signerinfo_issuerandserialnumber_serialnumber/` - PKI/X.509 signer info.
- `stringified_uuid_is_invalid/` - UUID validation.
- `systempromptsectioncache/` - system prompt section cache.
- `tengu_log_datadog_events/` - Datadog event logging.
- `undici_error_und_err_body_timeout/` - undici body timeout errors.
- `unsupported_platform/` - unsupported-platform / gitconfig handling.

## Related

- `../modules-manifest.json` - full module manifest.
- `../../../extracted/source/` - flat domain-based view.
