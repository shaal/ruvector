# ruvix-rpi-boot

Raspberry Pi 4/5 boot support for the RuVix Cognition Kernel (ADR-087 Phase D).

## Pi boot process

1. GPU ROM — loads `bootcode.bin` from SD card.
2. `bootcode.bin` — initializes SDRAM, loads `start4.elf`.
3. `start4.elf` — GPU firmware, parses `config.txt`, loads `kernel8.img`.
4. `kernel8.img` — RuVix kernel. Entry at `_start` with `x0 = DTB physical address`, `x1 = 0`, `x2 = 0`.

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-bcm2711`. Features: `rpi4` (-> `bcm2711/rpi4`), presumably `rpi5`.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
