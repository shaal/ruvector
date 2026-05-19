# ui/ruvocal/src/routes/api/transcribe/

Audio → text transcription endpoint.

## Files

- `+server.ts` — `POST` accepts an audio file from `lib/components/chat/VoiceRecorder.svelte`, calls the configured transcription provider, and returns the text. Subject to `lib/server/usageLimits.ts`.
