# Downloads only DR_grading/ (14.2 GB) of the 15.9 GB DDR repo into data/DDR/.
#
# Safe to re-run: finished files are skipped. It also retries by itself after
# a rate limit (HTTP 429) or an internet cut, so it can be left unattended.
# Set HF_TOKEN (free "Read" token from huggingface.co/settings/tokens) for
# much higher rate limits:  export HF_TOKEN=hf_...
import os
import sys
import time

from huggingface_hub import snapshot_download

MAX_ATTEMPTS = 50
WAIT_RATE_LIMIT = 300   # seconds — Hugging Face rate limits reset every 5 minutes
WAIT_NETWORK    = 60

if not os.environ.get("HF_TOKEN"):
    print("⚠️  HF_TOKEN not set — anonymous downloads get rate-limited (HTTP 429) after a few thousand files.")

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        snapshot_download(
            repo_id='ctmedtech/DDR-dataset',
            repo_type='dataset',
            local_dir='data/DDR',
            allow_patterns=['DR_grading/*'],
            max_workers=4,   # fewer parallel requests → fewer rate-limit hits
        )
        print("\n✅ Download complete.")
        break
    except Exception as e:   # rate limit or network cut (Ctrl+C still stops it)
        wait = WAIT_RATE_LIMIT if "429" in str(e) else WAIT_NETWORK
        print(f"\n[Attempt {attempt}/{MAX_ATTEMPTS}] {type(e).__name__}: {str(e)[:200]}")
        print(f"Waiting {wait}s, then resuming — files already downloaded are kept.")
        time.sleep(wait)
else:
    sys.exit("🚨 Download still failing after many attempts — check your internet, then re-run.")
