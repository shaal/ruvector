Small fixture models used by tests that need a real tokenizer/model artifact without depending on network access.

Files:
- `tokenizer.json` (~1.8MB) - HuggingFace-format tokenizer used by ruvLLM / training-adjacent tests.

Add new fixture artifacts here only if they are small enough to commit (lockfiles and large weights live elsewhere). Large model weights belong outside the repo or under `.gitignore`d paths.
