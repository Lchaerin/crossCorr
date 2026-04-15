#!/home/rllab/anaconda3/envs/hrtfSSL/bin/python3
"""
Download MRSAudio/MRSAudio dataset from HuggingFace.

실제 repo 구조:
  MRSAudio/MRSAudio
  └── MRSLife/
      ├── MRSDialogue/   ← HF repo 내 경로
      └── MRSSound/

저장 경로 (로컬):
  ./MRSAudio/MRSLife/       ← MRSLife 전체 (MRSDialogue + MRSSound)
  ./MRSAudio/MRSDialogue/   ← MRSDialogue만 별도 저장

Usage:
    python download_mrs_audio.py
    python download_mrs_audio.py --only MRSLife
    python download_mrs_audio.py --only MRSDialogue
    python download_mrs_audio.py --token hf_xxx    # private repo
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID   = "MRSAudio/MRSAudio"
BASE_DIR  = Path(__file__).parent / "MRSAudio"

# (local_subdir, allow_patterns)
TARGETS = {
    "MRSLife": {
        "local_dir":      BASE_DIR / "MRSLife",
        "allow_patterns": ["MRSLife/**"],
    },
    "MRSDialogue": {
        "local_dir":      BASE_DIR / "MRSDialogue",
        "allow_patterns": ["MRSLife/MRSDialogue/**"],
    },
}


def download(name: str, token: str | None = None) -> None:
    cfg = TARGETS[name]
    dest: Path = cfg["local_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] Downloading → {dest}")
    print(f"[{name}] patterns  : {cfg['allow_patterns']}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=cfg["allow_patterns"],
        token=token,
        local_dir_use_symlinks=False,
    )

    n = sum(1 for f in dest.rglob("*") if f.is_file())
    print(f"[{name}] Done  ({n} files)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=list(TARGETS),
        default=None,
        help="Download only this subset (default: both)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token",
    )
    args = parser.parse_args()

    targets = [args.only] if args.only else list(TARGETS)
    for name in targets:
        download(name, token=args.token)

    print("=== 완료 ===")
    for name in targets:
        dest = TARGETS[name]["local_dir"]
        n = sum(1 for f in dest.rglob("*") if f.is_file())
        print(f"  {dest}  ({n} files)")


if __name__ == "__main__":
    main()
