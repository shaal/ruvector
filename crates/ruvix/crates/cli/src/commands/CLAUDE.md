# ruvix-cli/src/commands

Subcommand implementations for the `ruvix` host-side CLI.

## Files

- `mod.rs` — re-exports the subcommand functions.
- `build.rs` — `ruvix build`: kernel image build wrapper (cargo + custom target spec).
- `flash.rs` — `ruvix flash`: write `kernel8.img` to a block device or SD card.
- `keys.rs` — `ruvix keys`: ML-DSA-65 signing key generation and management for secure boot.
- `config.rs` — `ruvix config`: read/edit kernel/RPi configuration.
- `dtb.rs` — `ruvix dtb`: device-tree validation / inspection.
- `monitor.rs` — `ruvix monitor`: serial console / log monitor.
- `security.rs` — `ruvix security`: secure-boot status + attestation checks.
