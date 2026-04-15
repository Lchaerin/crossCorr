#!/home/rllab/anaconda3/envs/hrtfSSL/bin/python3
"""
Download MRSAudio/MRSAudio - MRSSpeech subset from HuggingFace.

HF repo 구조:
  MRSAudio/MRSAudio
  └── MRSSpeech/
      ├── drama001/
      ├── drama002/
      └── ...

저장 경로 (로컬):
  ./MRSAudio/MRSSpeech/

Usage:
    python download_mrs_speech.py
    python download_mrs_speech.py --token hf_xxx    # private repo
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID  = "MRSAudio/MRSAudio"
DEST_DIR = Path(__file__).parent / "MRSAudio" / "MRSSpeech"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token",
    )
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[MRSSpeech] Downloading → {DEST_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DEST_DIR),
        allow_patterns=["MRSSpeech/**"],
        token=args.token,
        local_dir_use_symlinks=False,
    )

    n = sum(1 for f in DEST_DIR.rglob("*") if f.is_file())
    print(f"[MRSSpeech] Done  ({n} files)")
    print(f"  저장 경로: {DEST_DIR}")


if __name__ == "__main__":
    main()
