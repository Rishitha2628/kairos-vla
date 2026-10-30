"""Download SmolVLA base weights + the SmolVLM2 backbone's config/tokenizer (not its weights). Retries on flaky network."""
import time
from huggingface_hub import snapshot_download

JOBS = [
    ("lerobot/smolvla_base", None),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", ["*.json", "*.txt", "*.jinja", "tokenizer*"]),
]
for repo, patterns in JOBS:
    for attempt in range(1, 31):
        try:
            path = snapshot_download(repo, allow_patterns=patterns)
            print(f"OK {repo} -> {path}", flush=True)
            break
        except Exception as e:
            print(f"retry {attempt} {repo}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(3)
    else:
        raise SystemExit(f"FAILED {repo}")
