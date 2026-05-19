CI helper scripts for keeping lockfiles in sync and installing git hooks.

Files:
- `sync-lockfile.sh` - keeps `package-lock.json` aligned with `package.json` changes. Usable as a git hook, CI step, or manually.
- `ci-sync-lockfile.sh` - CI-friendly wrapper (no interactive prompts, fails on drift).
- `install-hooks.sh` - installs the project's git hooks from `../../.githooks/` into `.git/hooks/`.
