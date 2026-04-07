#!/usr/bin/env bash
# =============================================================================
# SLED v3 — Channel Ablation Study
# =============================================================================
# 각 config에 대해 B=1 순차 처리로 leytency + metrics를 동시에 측정한다.
# 결과 → stream_results.jsonl + 요약 테이블
#
# Configurations:
#   full        all channels  (L-mel, R-mel, ILD, IPD×2, HRTF-corr)
#   no_ild      remove ILD
#   no_ipd      remove IPD (sin+cos)
#   no_hrtf     remove HRTF cross-corr heatmap
#   no_binaural remove ALL binaural cues (ILD+IPD+HRTF) — only L/R mel
#
# Usage:
#   bash ablation.sh                          # eval-only using paths below
#   bash ablation.sh --train                  # train from scratch then eval
#   bash ablation.sh --configs "full no_hrtf" # run subset
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# =============================================================================
# ★  CHECKPOINT PATHS  ★  — set these before running
# =============================================================================
CKPT_full="/home/rllab/Desktop/crossCorr/checkpoints_archive/checkpoints_ver6_50db/sled_best.pt"
CKPT_no_ild="/home/rllab/Desktop/crossCorr/checkpoints_ver6_hrtfed_50db/sled_best.pt"
CKPT_no_ipd="/home/rllab/Desktop/crossCorr/checkpoints_ver6_noipd_50db/sled_best.pt"
CKPT_no_hrtf="/home/rllab/Desktop/crossCorr/checkpoints_ver6_nohrtf_50db/sled_best.pt"
CKPT_no_binaural="/home/rllab/Desktop/crossCorr/checkpoints_ver6_nobinaural_50db/sled_best.pt"
# =============================================================================

# ── 공통 설정 ─────────────────────────────────────────────────────────────────
DATASET_ROOT="./data_custom_hrtf"
SOFA_PATH="./hrtf/custom_mrs.sofa"
D_MODEL=256
N_CLASSES=209
DEVICE="cuda"
CONF_THRESH=0.35
RESULTS_FILE="stream_results.jsonl"
CONFIGS="full no_ild no_ipd no_hrtf no_binaural"

# ── 평가할 오디오 파일 목록 (B=1 순차 처리) ───────────────────────────────────
AUDIO_LIST=(
    "${DATASET_ROOT}/audio/test/scene_011000.wav"
    "${DATASET_ROOT}/audio/test/scene_011001.wav"
    "${DATASET_ROOT}/audio/test/scene_011002.wav"
    "${DATASET_ROOT}/audio/test/scene_011003.wav"
    "${DATASET_ROOT}/audio/test/scene_011004.wav"
    "${DATASET_ROOT}/audio/test/scene_011005.wav"
    "${DATASET_ROOT}/audio/test/scene_011006.wav"
    "${DATASET_ROOT}/audio/test/scene_011007.wav"
    "${DATASET_ROOT}/audio/test/scene_011008.wav"
    "${DATASET_ROOT}/audio/test/scene_011009.wav"
    "${DATASET_ROOT}/audio/test/scene_011010.wav"
    "${DATASET_ROOT}/audio/test/scene_011011.wav"
    "${DATASET_ROOT}/audio/test/scene_011012.wav"
    "${DATASET_ROOT}/audio/test/scene_011013.wav"
    "${DATASET_ROOT}/audio/test/scene_011014.wav"
    "${DATASET_ROOT}/audio/test/scene_011015.wav"
    "${DATASET_ROOT}/audio/test/scene_011016.wav"
    "${DATASET_ROOT}/audio/test/scene_011017.wav"
    "${DATASET_ROOT}/audio/test/scene_011018.wav"
    "${DATASET_ROOT}/audio/test/scene_011019.wav"
    "${DATASET_ROOT}/audio/test/scene_011020.wav"
    "${DATASET_ROOT}/audio/test/scene_011021.wav"
    "${DATASET_ROOT}/audio/test/scene_011022.wav"
    "${DATASET_ROOT}/audio/test/scene_011023.wav"
    "${DATASET_ROOT}/audio/test/scene_011024.wav"
    "${DATASET_ROOT}/audio/test/scene_011025.wav"
    "${DATASET_ROOT}/audio/test/scene_011026.wav"
    "${DATASET_ROOT}/audio/test/scene_011027.wav"
    "${DATASET_ROOT}/audio/test/scene_011028.wav"
    "${DATASET_ROOT}/audio/test/scene_011029.wav"
    "${DATASET_ROOT}/audio/test/scene_011030.wav"
    "${DATASET_ROOT}/audio/test/scene_011031.wav"
    "${DATASET_ROOT}/audio/test/scene_011032.wav"
    "${DATASET_ROOT}/audio/test/scene_011033.wav"
    "${DATASET_ROOT}/audio/test/scene_011034.wav"
    "${DATASET_ROOT}/audio/test/scene_011035.wav"
    "${DATASET_ROOT}/audio/test/scene_011036.wav"
    "${DATASET_ROOT}/audio/test/scene_011037.wav"
    "${DATASET_ROOT}/audio/test/scene_011038.wav"
    "${DATASET_ROOT}/audio/test/scene_011039.wav"
    "${DATASET_ROOT}/audio/test/scene_011040.wav"
    "${DATASET_ROOT}/audio/test/scene_011041.wav"
    "${DATASET_ROOT}/audio/test/scene_011042.wav"
    "${DATASET_ROOT}/audio/test/scene_011043.wav"
    "${DATASET_ROOT}/audio/test/scene_011044.wav"
    "${DATASET_ROOT}/audio/test/scene_011045.wav"
    "${DATASET_ROOT}/audio/test/scene_011046.wav"
    "${DATASET_ROOT}/audio/test/scene_011047.wav"
    "${DATASET_ROOT}/audio/test/scene_011048.wav"
    "${DATASET_ROOT}/audio/test/scene_011049.wav"
)
WINDOW_FRAMES=48    # 48 × 20ms = 960ms per window
N_WARMUP=10

# ── 학습 전용 설정 (--train 없으면 무시) ──────────────────────────────────────
EPOCHS=200
LR=1e-4
TRAIN_WINDOW_FRAMES=256
MIN_LOUDNESS_DB=-60.0
BASE_CKPT_DIR="./checkpoints/ablation"
BASE_LOG_DIR="./runs/ablation"

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
DO_TRAIN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --train)              DO_TRAIN=true;                shift ;;
        --dataset-root)       DATASET_ROOT="$2";            shift 2 ;;
        --sofa-path)          SOFA_PATH="$2";               shift 2 ;;
        --epochs)             EPOCHS="$2";                  shift 2 ;;
        --lr)                 LR="$2";                      shift 2 ;;
        --d-model)            D_MODEL="$2";                 shift 2 ;;
        --n-classes)          N_CLASSES="$2";               shift 2 ;;
        --device)             DEVICE="$2";                  shift 2 ;;
        --window-frames)      WINDOW_FRAMES="$2";           shift 2 ;;
        --min-loudness-db)    MIN_LOUDNESS_DB="$2";         shift 2 ;;
        --conf-thresh)        CONF_THRESH="$2";             shift 2 ;;
        --configs)            CONFIGS="$2";                 shift 2 ;;
        --results-file)       RESULTS_FILE="$2";            shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

config_train_flags() {
    case $1 in
        full)        echo "" ;;
        no_ild)      echo "--no-ild" ;;
        no_ipd)      echo "--no-ipd" ;;
        no_hrtf)     echo "--no-hrtf-corr" ;;
        no_binaural) echo "--no-ild --no-ipd --no-hrtf-corr" ;;
        *) echo "[ERROR] Unknown config: $1" >&2; exit 1 ;;
    esac
}

resolve_ckpt() {
    local cfg="$1"
    local varname="CKPT_${cfg}"
    local explicit="${!varname:-}"
    if [[ -n "$explicit" ]]; then
        echo "$explicit"
    else
        echo "${BASE_CKPT_DIR}/${cfg}/sled_best.pt"
    fi
}

# ── 시작 안내 ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  SLED v3 Channel Ablation Study"
echo "  Configs  : $CONFIGS"
echo "  Train    : $DO_TRAIN"
if [[ "$DO_TRAIN" == "true" ]]; then
    echo "  Epochs   : $EPOCHS"
fi
echo "  Files    : ${#AUDIO_LIST[@]} files (${AUDIO_LIST[0]##*/} …)"
echo "  Window   : ${WINDOW_FRAMES} frames × 20ms = $((WINDOW_FRAMES * 20))ms"
echo "  Results  : $RESULTS_FILE"
echo "============================================================"
echo ""
echo "  Checkpoint paths:"
for CFG in $CONFIGS; do
    echo "    ${CFG}: $(resolve_ckpt "$CFG")"
done
echo ""

> "$RESULTS_FILE"   # 이전 결과 초기화

# ── 존재하는 파일만 필터링 ────────────────────────────────────────────────────
EXISTING_AUDIO=()
for f in "${AUDIO_LIST[@]}"; do
    [[ -f "$f" ]] && EXISTING_AUDIO+=("$f")
done

if [[ ${#EXISTING_AUDIO[@]} -eq 0 ]]; then
    echo "[ERROR] No audio files found. Check DATASET_ROOT and AUDIO_LIST."
    exit 1
fi
echo "  Found ${#EXISTING_AUDIO[@]} / ${#AUDIO_LIST[@]} audio files."
echo ""

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
for CONFIG in $CONFIGS; do
    BEST_CKPT=$(resolve_ckpt "$CONFIG")
    EXTRA_FLAGS=$(config_train_flags "$CONFIG")

    echo "------------------------------------------------------------"
    log "Config: ${CONFIG}"
    echo "------------------------------------------------------------"

    # ── 학습 (선택) ───────────────────────────────────────────────────────────
    if [[ "$DO_TRAIN" == "true" ]]; then
        CKPT_DIR="${BASE_CKPT_DIR}/${CONFIG}"
        LOG_DIR="${BASE_LOG_DIR}/${CONFIG}"
        mkdir -p "$CKPT_DIR" "$LOG_DIR"

        RESUME_FLAG=""
        if [[ -f "$BEST_CKPT" ]]; then
            log "Resuming from: $BEST_CKPT"
            RESUME_FLAG="--resume $BEST_CKPT"
        fi

        log "Training …"
        python -m sled.train \
            --dataset-root    "$DATASET_ROOT"       \
            --sofa-path       "$SOFA_PATH"          \
            --epochs          "$EPOCHS"             \
            --lr              "$LR"                 \
            --d-model         "$D_MODEL"            \
            --n-classes       "$N_CLASSES"          \
            --device          "$DEVICE"             \
            --window-frames   "$TRAIN_WINDOW_FRAMES" \
            --min-loudness-db "$MIN_LOUDNESS_DB"    \
            --checkpoint-dir  "$CKPT_DIR"           \
            --log-dir         "$LOG_DIR"            \
            $RESUME_FLAG                            \
            $EXTRA_FLAGS

        log "Training done."
    fi

    # ── 체크포인트 없으면 스킵 ────────────────────────────────────────────────
    if [[ ! -f "$BEST_CKPT" ]]; then
        log "WARNING: checkpoint not found at ${BEST_CKPT} — skipping"
        echo ""
        continue
    fi

    # ── B=1 순차 평가 (latency + metrics) ─────────────────────────────────────
    log "Evaluating (B=1, ${#EXISTING_AUDIO[@]} files): $BEST_CKPT"
    python -m sled.stream_bench \
        --ckpt           "$BEST_CKPT"            \
        --audio          "${EXISTING_AUDIO[@]}"  \
        --sofa-path      "$SOFA_PATH"            \
        --window-frames  "$WINDOW_FRAMES"        \
        --conf-thresh    "$CONF_THRESH"          \
        --d-model        "$D_MODEL"              \
        --n-classes      "$N_CLASSES"            \
        --device         "$DEVICE"               \
        --n-warmup       "$N_WARMUP"             \
        --output-json    "$RESULTS_FILE"

    log "Done: ${CONFIG}"
    echo ""
done

# ── 요약 테이블 ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Ablation Results (B=1 streaming, ${#EXISTING_AUDIO[@]} files)"
echo "  window = ${WINDOW_FRAMES} frames × 20ms = $((WINDOW_FRAMES * 20))ms audio"
echo "============================================================"
python3 - "$RESULTS_FILE" "$WINDOW_FRAMES" <<'PYEOF'
import json, sys, math

path          = sys.argv[1]
window_frames = int(sys.argv[2])
window_ms     = window_frames * 20

rows = []
try:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
except FileNotFoundError:
    print(f"[ERROR] {path} not found"); sys.exit(1)

if not rows:
    print("No results."); sys.exit(0)

CONFIG_ORDER = ['full', 'no_ild', 'no_ipd', 'no_hrtf', 'no_binaural']

def config_name(r):
    ild, ipd, hrtf = r['use_ild'], r['use_ipd'], r['use_hrtf_corr']
    if     ild and     ipd and     hrtf: return 'full'
    if not ild and     ipd and     hrtf: return 'no_ild'
    if     ild and not ipd and     hrtf: return 'no_ipd'
    if     ild and     ipd and not hrtf: return 'no_hrtf'
    if not ild and not ipd and not hrtf: return 'no_binaural'
    return f'custom(ild={ild},ipd={ipd},hrtf={hrtf})'

rows.sort(key=lambda r: CONFIG_ORDER.index(config_name(r))
          if config_name(r) in CONFIG_ORDER else 99)

full = next((r for r in rows if config_name(r) == 'full'), None)

def fmt(v, spec):
    return f'{v:{spec}}' if v is not None else '-'.center(len(f'{0:{spec}}'))

hdr = (f"  {'Config':<14} {'ILD':^3} {'IPD':^3} {'HRTF':^4}  "
       f"{'Loss':>7}  {'ΔLoss':>7}  {'Det-F1':>6}  {'Cls-Acc':>7}  {'DOA-MAE':>7}  "
       f"{'avg/win':>8}  {'p99/win':>8}  {'RTF':>8}")
sep = "  " + "-" * (len(hdr) - 2)
print(hdr)
print(sep)

for r in rows:
    name  = config_name(r)
    loss  = r['loss']
    dloss = (loss - full['loss']) if full and name != 'full' and loss is not None and full['loss'] is not None else math.nan

    print(
        f"  {name:<14} "
        f"{'Y' if r['use_ild'] else 'N':^3} "
        f"{'Y' if r['use_ipd'] else 'N':^3} "
        f"{'Y' if r['use_hrtf_corr'] else 'N':^4}  "
        f"{fmt(r['loss'],        '7.4f')}  "
        f"{(('—' if math.isnan(dloss) else ('+' if dloss>=0 else '')+f'{dloss:.4f}')):>7}  "
        f"{fmt(r['det_f1'],      '6.4f')}  "
        f"{fmt(r['cls_acc'],     '7.4f')}  "
        f"{fmt(r['doa_mae_deg'], '6.2f')}°  "
        f"{r['avg_window_ms']:>7.2f}ms  "
        f"{r['p99_window_ms']:>7.2f}ms  "
        f"{r['rtf']:>8.5f}"
    )

print(sep)
print(f"  ΔLoss = config Loss − full Loss  (양수 = 성능 하락)")
print(f"  RTF: Real-Time Factor = inference time / audio time  (낮을수록 빠름)")
PYEOF

echo ""
log "Results saved to: ${RESULTS_FILE}"
