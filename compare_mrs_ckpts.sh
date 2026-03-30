#!/usr/bin/env bash
# =============================================================================
# SLED v3 — MRS 체크포인트 비교 (checkpoints_mrs_mix_ft vs checkpoints_mrs_mix2)
# data_mrs_mix test split 50개 씬으로 Det-F1 / Cls-Acc / DOA-MAE / 레이턴시 비교
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# =============================================================================
# ★  비교할 체크포인트  ★
# =============================================================================
CKPT_ft="./checkpoints_mrs_mix_ft/sled_best.pt"
CKPT_2="./checkpoints_mrs_mix2/sled_best.pt"
# =============================================================================

DATASET_ROOT="./data_mrs_mix"
SOFA_PATH="./hrtf/p0001.sofa"
N_CLASSES=209
DEVICE="cuda"
CONF_THRESH=0.35
WINDOW_FRAMES=48
N_WARMUP=5
RESULTS_FILE="compare_mrs_results.jsonl"

# ── test 씬 50개 선택 (scene_006750 ~ scene_006799) ──────────────────────────
AUDIO_LIST=()
for i in $(seq 6750 6799); do
    f="${DATASET_ROOT}/audio/test/scene_$(printf '%06d' $i).wav"
    [[ -f "$f" ]] && AUDIO_LIST+=("$f")
done

echo "============================================================"
echo "  SLED v3 MRS Checkpoint Comparison"
echo "  ft  : $CKPT_ft"
echo "  mix2: $CKPT_2"
echo "  Files: ${#AUDIO_LIST[@]} test scenes"
echo "  Window: ${WINDOW_FRAMES} frames × 20ms = $((WINDOW_FRAMES * 20))ms"
echo "  Results: $RESULTS_FILE"
echo "============================================================"
echo ""

if [[ ${#AUDIO_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] test 씬 파일이 없습니다. DATASET_ROOT를 확인하세요."
    exit 1
fi

> "$RESULTS_FILE"

# ── 평가 함수 ─────────────────────────────────────────────────────────────────
run_eval() {
    local label="$1"
    local ckpt="$2"

    if [[ ! -f "$ckpt" ]]; then
        echo "[SKIP] $label: 체크포인트 없음 ($ckpt)"
        return
    fi

    echo "------------------------------------------------------------"
    echo "[EVAL] $label"
    echo "       ckpt: $ckpt"
    echo "------------------------------------------------------------"

    python -m sled.stream_bench \
        --ckpt          "$ckpt"              \
        --audio         "${AUDIO_LIST[@]}"   \
        --sofa-path     "$SOFA_PATH"         \
        --window-frames "$WINDOW_FRAMES"     \
        --conf-thresh   "$CONF_THRESH"       \
        --n-classes     "$N_CLASSES"         \
        --device        "$DEVICE"            \
        --n-warmup      "$N_WARMUP"          \
        --output-json   "$RESULTS_FILE"

    echo ""
}

run_eval "mrs_mix_ft  (finetune)" "$CKPT_ft"
run_eval "mrs_mix2    (scratch2)" "$CKPT_2"

# ── 요약 테이블 ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Comparison Results  (${#AUDIO_LIST[@]} test scenes)"
echo "============================================================"

python3 - "$RESULTS_FILE" <<'PYEOF'
import json, sys, math

path = sys.argv[1]
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
    print("결과 없음."); sys.exit(0)

# 체크포인트 경로에서 이름 추출
def ckpt_label(r):
    p = r.get('ckpt', '')
    parts = p.replace('\\', '/').split('/')
    # checkpoints_XXX 디렉터리 이름 찾기
    for part in parts:
        if part.startswith('checkpoints_mrs'):
            return part
    return parts[-2] if len(parts) >= 2 else p

labels = [ckpt_label(r) for r in rows]
ref = rows[0]  # 첫 번째가 기준

def fmt(v, spec):
    if v is None: return '-'.rjust(len(f'{0:{spec}}'))
    return f'{v:{spec}}'

def delta(a, b, higher_better=True):
    if a is None or b is None: return '    -'
    d = b - a
    sign = '+' if d >= 0 else ''
    better = (d > 0) == higher_better
    tag = '↑' if better else '↓'
    return f'{sign}{d:+.4f}{tag}'

metrics = [
    ('loss',        'Loss',     '7.4f', False),
    ('det_f1',      'Det-F1',   '6.4f', True),
    ('cls_acc',     'Cls-Acc',  '7.4f', True),
    ('doa_mae_deg', 'DOA-MAE',  '6.2f', False),
]

# 헤더
name_w = max(len(ckpt_label(r)) for r in rows) + 2
hdr = f"  {'Checkpoint':<{name_w}}"
for _, mname, spec, _ in metrics:
    w = max(len(mname), len(f'{0:{spec}}')) + 2
    hdr += f"  {mname:^{w}}"
hdr += f"  {'avg/win':>9}  {'p99/win':>9}  {'RTF':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for i, r in enumerate(rows):
    name = ckpt_label(r)
    line = f"  {name:<{name_w}}"
    for key, mname, spec, higher_better in metrics:
        v = r.get(key)
        cell = fmt(v, spec)
        if i > 0:
            cell += f" ({delta(ref.get(key), v, higher_better)})"
        w = max(len(mname), len(f'{0:{spec}}')) + 2
        line += f"  {cell:^{w + (10 if i > 0 else 0)}}"
    line += f"  {r.get('avg_window_ms', 0):>8.2f}ms"
    line += f"  {r.get('p99_window_ms', 0):>8.2f}ms"
    line += f"  {r.get('rtf', 0):>8.5f}"
    print(line)

print("  " + "-" * (len(hdr) - 2))
print("  ↑ = 기준 대비 향상  ↓ = 기준 대비 하락  (기준: 첫 번째 체크포인트)")
print("  DOA-MAE: 낮을수록 좋음  /  RTF: 낮을수록 빠름")
PYEOF

echo ""
echo "결과 저장: ${RESULTS_FILE}"
