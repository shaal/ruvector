# ruos-thermal/deploy

Deployment artifacts for installing `ruos-thermal` on a Raspberry Pi 5 (ADR-174).

## Files

- `install.sh` — Installer script: copies the binary, registers the unit/timer.
- `ruos-thermal.service` — systemd unit definition.
- `ruos-thermal.timer` — systemd timer (periodic invocation; aligns with the future 5-second supervisor tick).

These ship a one-shot read-only thermal probe today; the long-running daemon arrives in iter 92-97.
