# ruvector-mmwave/src

- `lib.rs` — entire crate in one file. `#![cfg_attr(not(any(test, feature = "std")), no_std)]`. Defines `MAX_PAYLOAD = 64`, the `Event` enum (`Breathing`, `HeartRate`, `Distance`, `Presence`, `Unknown { frame_type: u16, payload_len: u16 }`, `ChecksumError`, `Resync`), the internal `State` machine (`Sof` → `Header` → `Payload` → `Trailer`), checksum verification, and the public `Parser`.

See `../CLAUDE.md`.
