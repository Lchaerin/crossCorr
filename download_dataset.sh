#!/usr/bin/env bash
# =============================================================================
# SLED Dataset Download Script
# =============================================================================
# Downloads all external data sources needed to run sled/dataset/build_dataset.py
#
# Required sources:
#   [1] FSD50K  — ~24 GB, 200-class sound effects (Zenodo 4060432)
#
# Optional (currently not used — synthesizer uses direct HRTF convolution):
#   [2] TAU-SRIR_DB   — room impulse responses (Zenodo 6408612)
#   [3] TAU-SNoise_DB — ambient noise recordings (Zenodo 6408612)
#
# HRTF:
#   hrtf/p0001.sofa already present — no download needed.
#
# Usage:
#   bash download_dataset.sh                # download everything (FSD50K)
#   bash download_dataset.sh --fsd50k-only  # same
#   bash download_dataset.sh --tau          # also download TAU-SRIR/SNoise
#   bash download_dataset.sh --check        # verify existing downloads only
#
# After download, run:
#   python -m sled.dataset.build_dataset --num-train 10000 --num-val 1000 --num-test 500
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFX_DIR="${SCRIPT_DIR}/soud_effects"
HRTF_DIR="${SCRIPT_DIR}/hrtf"
SOURCES_DIR="${SCRIPT_DIR}/sources"
FSD50K_DIR="${SOURCES_DIR}/FSD50K"
TAU_DIR="${SOURCES_DIR}/TAU-SRIR"

DOWNLOAD_TAU=false
CHECK_ONLY=false

# Parse args
for arg in "$@"; do
  case "$arg" in
    --tau)          DOWNLOAD_TAU=true ;;
    --check)        CHECK_ONLY=true ;;
    --fsd50k-only)  ;;  # default behaviour
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERR]${NC}   $*"; }
section() { echo -e "\n${GREEN}══ $* ══${NC}"; }

need_cmd() {
  if ! command -v "$1" &>/dev/null; then
    err "Required command not found: $1"
    echo "  Install with: sudo apt-get install $2"
    exit 1
  fi
}

# Prefer aria2c (parallel) over wget, fallback to curl
download() {
  local url="$1" dest="$2"
  if command -v aria2c &>/dev/null; then
    aria2c --continue=true --max-connection-per-server=8 --split=8 \
           --dir="$(dirname "$dest")" --out="$(basename "$dest")" "$url"
  elif command -v wget &>/dev/null; then
    wget --continue --show-progress -O "$dest" "$url"
  else
    curl -L --continue-at - --progress-bar -o "$dest" "$url"
  fi
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
section "Pre-flight"

need_cmd python3 python3
need_cmd unzip  unzip

# Check for 7z (needed for FSD50K split-zip extraction)
if ! command -v 7z &>/dev/null; then
  warn "7z not found — required for FSD50K split-zip extraction."
  warn "Install with: sudo apt-get install p7zip-full"
  warn "(continuing; will abort at extraction step if missing)"
fi

# Check HRTF
if [[ -f "${HRTF_DIR}/p0001.sofa" ]]; then
  info "HRTF: ${HRTF_DIR}/p0001.sofa ✓"
else
  warn "HRTF not found at ${HRTF_DIR}/p0001.sofa"
  warn "Download SONICOM HRTF manually:"
  warn "  https://www.sonicom.eu/wp-content/uploads/2022/06/p0001.sofa"
fi

if [[ "$CHECK_ONLY" == true ]]; then
  section "Check mode — no downloads"
fi

# ── FSD50K ────────────────────────────────────────────────────────────────────
section "FSD50K  (Freesound Dataset 50K)"
# Zenodo record: https://zenodo.org/records/4060432
# ~24 GB total: dev (~19 GB, 5 zips) + eval (~4.7 GB, 1 zip) + metadata
FSD50K_BASE="https://zenodo.org/records/4060432/files"
mkdir -p "$FSD50K_DIR"

FSD50K_DEV_PARTS=(
  "FSD50K.dev_audio.z01"
  "FSD50K.dev_audio.z02"
  "FSD50K.dev_audio.z03"
  "FSD50K.dev_audio.z04"
  "FSD50K.dev_audio.z05"
  "FSD50K.dev_audio.zip"
)
FSD50K_OTHER_FILES=(
  "FSD50K.eval_audio.zip"
  "FSD50K.metadata.zip"
  "FSD50K.doc.zip"
)

if [[ "$CHECK_ONLY" == false ]]; then
  info "Downloading FSD50K to ${FSD50K_DIR} ..."

  # Dev audio parts (.z01–.z05 + .zip)
  for fname in "${FSD50K_DEV_PARTS[@]}"; do
    dest="${FSD50K_DIR}/${fname}"
    if [[ -f "$dest" ]]; then
      info "  Already exists: ${fname}"
    else
      info "  Downloading: ${fname}"
      download "${FSD50K_BASE}/${fname}?download=1" "$dest"
    fi
  done

  # Other files (eval, metadata, doc)
  for fname in "${FSD50K_OTHER_FILES[@]}"; do
    dest="${FSD50K_DIR}/${fname}"
    if [[ -f "$dest" ]]; then
      info "  Already exists: ${fname}"
    else
      info "  Downloading: ${fname}"
      download "${FSD50K_BASE}/${fname}?download=1" "$dest"
    fi
  done

  # Extract dev audio (split zip)
  # FSD50K uses Windows-style split zip: central directory offsets are relative
  # to the last disk, not the combined file — only 7z handles this correctly.
  if [[ ! -d "${FSD50K_DIR}/FSD50K.dev_audio" ]]; then
    info "Extracting dev audio (split zip via 7z) ..."
    if ! command -v 7z &>/dev/null; then
      err "7z is required but not installed. Run: sudo apt-get install p7zip-full"
      exit 1
    fi
    # Point 7z at the last part (.zip); it auto-locates .z01-.z05 in the same dir
    7z x "${FSD50K_DIR}/FSD50K.dev_audio.zip" -o"${FSD50K_DIR}" -y
  else
    info "  Dev audio already extracted."
  fi

  # Extract eval audio
  if [[ ! -d "${FSD50K_DIR}/FSD50K.eval_audio" ]]; then
    info "Extracting eval audio ..."
    unzip -q "${FSD50K_DIR}/FSD50K.eval_audio.zip" -d "$FSD50K_DIR"
  else
    info "  Eval audio already extracted."
  fi

  # Extract metadata
  if [[ ! -d "${FSD50K_DIR}/FSD50K.metadata" ]]; then
    info "Extracting metadata ..."
    unzip -q "${FSD50K_DIR}/FSD50K.metadata.zip" -d "$FSD50K_DIR"
  else
    info "  Metadata already extracted."
  fi
fi

# Count dev + eval clips
DEV_COUNT=$(find "${FSD50K_DIR}/FSD50K.dev_audio"  -name "*.wav" 2>/dev/null | wc -l || echo 0)
EVAL_COUNT=$(find "${FSD50K_DIR}/FSD50K.eval_audio" -name "*.wav" 2>/dev/null | wc -l || echo 0)
info "FSD50K: dev=${DEV_COUNT} clips, eval=${EVAL_COUNT} clips"

# ── Link FSD50K into soud_effects ─────────────────────────────────────────────
# The synthesizer reads from soud_effects/*.{mp3,wav}.
# We copy (or link) the class_map-aware directory structure.
# Strategy: use the vocabulary.csv to group clips by class,
#           then populate soud_effects/ with symlinks organised by class label.

if [[ "$CHECK_ONLY" == false && ( "$DEV_COUNT" -gt 0 || "$EVAL_COUNT" -gt 0 ) ]]; then
  section "Linking FSD50K clips into soud_effects/"
  # Actual FSD50K layout: ground_truth is a top-level sibling dir, not under metadata
  VOCAB="${FSD50K_DIR}/FSD50K.ground_truth/vocabulary.csv"
  DEV_GT="${FSD50K_DIR}/FSD50K.ground_truth/dev.csv"
  EVAL_GT="${FSD50K_DIR}/FSD50K.ground_truth/eval.csv"

  if [[ -f "$VOCAB" && -f "$DEV_GT" ]]; then
    info "Populating ${SFX_DIR}/ via symlinks ..."
    mkdir -p "$SFX_DIR"

    /home/rllab/anaconda3/bin/python3 - <<'PYEOF'
import csv, os, sys

script_dir = os.environ.get("SCRIPT_DIR", ".")
fsd50k_dir = os.path.join(script_dir, "sources", "FSD50K")
sfx_dir    = os.path.join(script_dir, "soud_effects")

# Actual FSD50K layout: vocabulary.csv in FSD50K.ground_truth/
# Format: index,label,mid  (no header)
vocab_path   = os.path.join(fsd50k_dir, "FSD50K.ground_truth", "vocabulary.csv")
dev_gt_path  = os.path.join(fsd50k_dir, "FSD50K.ground_truth", "dev.csv")
eval_gt_path = os.path.join(fsd50k_dir, "FSD50K.ground_truth", "eval.csv")
dev_audio    = os.path.join(fsd50k_dir, "FSD50K.dev_audio")
eval_audio   = os.path.join(fsd50k_dir, "FSD50K.eval_audio")

if not os.path.exists(vocab_path):
    print(f"  [SKIP] {vocab_path} not found — skipping symlink creation")
    sys.exit(0)

# vocabulary.csv: index,label,mid  (no header row)
id2label = {}
with open(vocab_path) as f:
    for line in f:
        parts = line.strip().split(",", 2)
        if len(parts) == 3:
            _, label, mid = parts
            id2label[mid.strip()] = label.strip().replace(" ", "_").replace("/", "-")

def link_csv(gt_path, audio_dir):
    if not os.path.exists(gt_path):
        return 0
    linked = 0
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname  = str(row.get("fname", "")).strip()
            mids   = row.get("mids", "").strip().split(",")
            # Use first (primary) mid
            label  = id2label.get(mids[0].strip(), "unknown") if mids else "unknown"
            src    = os.path.join(audio_dir, f"{fname}.wav")
            if not os.path.exists(src):
                continue
            dst_dir = os.path.join(sfx_dir, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{fname}.wav")
            if not os.path.exists(dst):
                os.symlink(src, dst)
            linked += 1
    return linked

n_dev  = link_csv(dev_gt_path,  dev_audio)
n_eval = link_csv(eval_gt_path, eval_audio)
print(f"  Linked {n_dev} dev + {n_eval} eval clips into {sfx_dir}/")
print(f"  Classes with clips: {len(os.listdir(sfx_dir))}")
PYEOF
    SCRIPT_DIR="$SCRIPT_DIR" /home/rllab/anaconda3/bin/python3 - <<'PYEOF2'
import os, sys

sfx_dir = os.path.join(os.environ.get("SCRIPT_DIR", "."), "soud_effects")
if not os.path.isdir(sfx_dir):
    sys.exit(0)
classes = [d for d in os.listdir(sfx_dir) if os.path.isdir(os.path.join(sfx_dir, d))]
print(f"  soud_effects/ has {len(classes)} class subdirs")
PYEOF2
  else
    warn "FSD50K metadata not found — cannot create class symlinks."
    warn "You can manually copy wav files to ${SFX_DIR}/"
  fi
fi

# ── TAU-SRIR (optional) ───────────────────────────────────────────────────────
if [[ "$DOWNLOAD_TAU" == true ]]; then
  section "TAU-SRIR_DB + TAU-SNoise_DB  (optional)"
  # Zenodo record: https://zenodo.org/records/6408612
  TAU_BASE="https://zenodo.org/records/6408612/files"
  mkdir -p "$TAU_DIR"

  # Room names
  ROOMS=(bomb_shelter gym pb132 pc226 sa203 sc203 se203 tb103 tc352)

  if [[ "$CHECK_ONLY" == false ]]; then
    info "Downloading TAU-SRIR_DB ..."
    for room in "${ROOMS[@]}"; do
      fname="TAU-SRIR_DB_${room}.zip"
      dest="${TAU_DIR}/${fname}"
      if [[ -f "$dest" ]]; then
        info "  Already exists: ${fname}"
      else
        info "  Downloading: ${fname}"
        download "${TAU_BASE}/${fname}?download=1" "$dest"
      fi
      if [[ ! -d "${TAU_DIR}/${room}" ]]; then
        unzip -q "$dest" -d "$TAU_DIR"
      fi
    done

    info "Downloading TAU-SNoise_DB ..."
    SNOISE_ZIP="${TAU_DIR}/TAU-SNoise_DB.zip"
    if [[ ! -f "$SNOISE_ZIP" ]]; then
      download "${TAU_BASE}/TAU-SNoise_DB.zip?download=1" "$SNOISE_ZIP"
    fi
    if [[ ! -d "${TAU_DIR}/TAU-SNoise_DB" ]]; then
      unzip -q "$SNOISE_ZIP" -d "$TAU_DIR"
    fi
  else
    for room in "${ROOMS[@]}"; do
      if [[ -d "${TAU_DIR}/${room}" ]]; then
        info "  TAU-SRIR ${room} ✓"
      else
        warn "  TAU-SRIR ${room} missing"
      fi
    done
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "Summary"

check_item() {
  local label="$1" path="$2" count_cmd="$3"
  if [[ -e "$path" ]]; then
    local n
    n=$(eval "$count_cmd" 2>/dev/null || echo "?")
    info "  ✓ ${label}: ${n}"
  else
    warn "  ✗ ${label}: not found at ${path}"
  fi
}

check_item "HRTF p0001.sofa" \
  "${HRTF_DIR}/p0001.sofa" \
  "echo 'present'"

check_item "FSD50K dev audio" \
  "${FSD50K_DIR}/FSD50K.dev_audio" \
  "find '${FSD50K_DIR}/FSD50K.dev_audio' -name '*.wav' | wc -l | tr -d ' '; echo ' clips'"

check_item "FSD50K eval audio" \
  "${FSD50K_DIR}/FSD50K.eval_audio" \
  "find '${FSD50K_DIR}/FSD50K.eval_audio' -name '*.wav' | wc -l | tr -d ' '; echo ' clips'"

check_item "soud_effects (linked)" \
  "${SFX_DIR}" \
  "find '${SFX_DIR}' -name '*.wav' -o -name '*.mp3' 2>/dev/null | wc -l | tr -d ' '; echo ' clips'"

if [[ "$DOWNLOAD_TAU" == true ]]; then
  for room in bomb_shelter gym pb132 pc226 sa203 sc203 se203 tb103 tc352; do
    check_item "TAU-SRIR ${room}" \
      "${TAU_DIR}/${room}" \
      "echo 'present'"
  done
fi

echo ""
info "Next step:"
info "  python -m sled.dataset.build_dataset \\"
info "    --num-train 10000 --num-val 1000 --num-test 500 \\"
info "    --output-dir ./data --workers 16"
