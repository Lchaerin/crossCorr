# SLED v3.1 → v3.2 Revision Guide (Over-provisioned Slots + NMS + 학습 전략)

> **선행 조건**: revise_ver3.md의 수정 사항이 모두 적용된 상태에서 진행.
> **목적**: 슬롯 수를 3→12로 늘리고 추론 시 NMS로 필터링하여 다중 음원 DOA 성능 개선.
> 학습 전략(커리큘럼, 데이터)과 moderate scaling도 포함.

---

## 수정 개요

| # | 수정 사항 | 파일 | 난이도 |
|---|----------|------|--------|
| 1 | n_slots 3→12 (over-provisioned slots) | `sled.py`, `decoder.py` | 하 |
| 2 | Hungarian matching 비대칭 지원 (12 pred × 3 GT) | `train.py` | 중 |
| 3 | Confidence loss 밸런싱 (positive weight boost) | `train.py` | 하 |
| 4 | DOA-aware NMS (추론 시) | 신규 `nms.py` | 하 |
| 5 | SLEDv3.forward()에 NMS 적용 경로 추가 | `sled.py` | 중 |
| 6 | Single→Multi 커리큘럼 학습 | `train.py` | 중 |
| 7 | Moderate scaling (d_model, conformer) | `sled.py`, `train.py` | 하 |

---

## 수정 1: n_slots 3→12

### 개념
기존 3슬롯은 정확히 GT 수만큼만 예측해야 하므로 학습 초기 collapse가 쉽다.
12슬롯으로 늘리면 각 슬롯이 자유롭게 후보를 제안하고, 비매칭 슬롯은
confidence=0으로 억제된다. 추론 시 NMS로 중복을 제거.

### 변경 내용

#### `sled.py` — `SLEDv3.__init__()`

**변경 전 (line 52-54):**
```python
def __init__(self, sofa_path: str, d_model: int = 256,
             n_slots: int = 3, n_classes: int = 209,
             ...):
```

**변경 후:**
```python
def __init__(self, sofa_path: str, d_model: int = 256,
             n_slots: int = 12, n_classes: int = 209,
             max_sources: int = 3,
             ...):
```

`max_sources`는 GT의 최대 음원 수를 의미. NMS 후 최종 출력 상한으로 사용.

```python
self.n_slots      = n_slots       # 12 (over-provisioned)
self.max_sources  = max_sources   # 3 (GT 최대 음원 수)
```

#### `decoder.py` — `CrossAttentionQuerySelector.__init__()`

`n_slots` 파라미터만 12로 바뀌면 자동으로 slot_queries가 [12, d_model]로 생성됨.
revise_ver3의 spatial prior 초기화도 12개 슬롯에 맞게 30° 간격으로 변경:

```python
for s in range(n_slots):
    angle = 2 * math.pi * s / n_slots   # 12 slots → 30° 간격
    slot_init[s, 0] += 0.5 * math.cos(angle)
    slot_init[s, 1] += 0.5 * math.sin(angle)
```

#### `decoder.py` — `IterativeRefinementDecoder`

`n_slots` 파라미터가 12로 바뀌면 DOA refinement가 12 matching queries에 적용됨.
코드 변경 불필요 — `__init__`의 `n_slots` 인자가 그대로 전파됨.

#### `heads.py` — `DetectionHeads.__init__()`

`n_slots=12`로 변경. 코드 구조 변경 없음 — 인자만 바뀜.

---

## 수정 2: Hungarian Matching 비대칭 지원

### 문제
현재 cost matrix는 `[BT, S, S]`로 pred 슬롯 수 = GT 슬롯 수를 가정.
12 pred × 3 GT가 되면 cost matrix가 `[BT, 12, 3]`이 되어야 함.

### 변경 내용

#### `train.py` — `_compute_single_layer_loss()`

GT의 S 차원은 여전히 `max_sources=3`. Pred의 S 차원은 `n_slots=12`.
cost matrix를 `[BT, 12, 3]`으로 구성하도록 수정.

**변경 전 (line 112-163):**
```python
B, T, S, C = class_logits.shape
# ... (S가 pred와 GT 모두 같다고 가정)
cls_gt_exp = cls_gt_flat.clamp(0, C - 1).unsqueeze(1).expand(BT, S, S)
cls_cost   = -prob.gather(2, cls_gt_exp)                  # [BT, S, S]
doa_cost   = 1.0 - torch.bmm(pred_norm, gt_norm.transpose(1, 2))  # [BT, S, S]
cost_np    = (cls_cost + 0.5 * doa_cost).cpu().numpy()   # [BT, S, S]
```

**변경 후:**
```python
B, T, S_pred, C = class_logits.shape
S_gt = gt_cls.shape[2]   # max_sources (3)

# ... flatten 시 pred와 GT 분리
logits_flat = class_logits.reshape(BT, S_pred, C)    # [BT, 12, C]
doa_flat    = doa_vec.reshape(BT, S_pred, 3)         # [BT, 12, 3]
conf_flat   = confidence.reshape(BT, S_pred)         # [BT, 12]
cls_gt_flat = gt_cls.reshape(BT, S_gt)               # [BT, 3]
doa_gt_flat = gt_doa.reshape(BT, S_gt, 3)            # [BT, 3, 3]
mask_flat   = gt_mask.reshape(BT, S_gt)              # [BT, 3]

with torch.no_grad():
    prob = logits_flat.softmax(-1)                                    # [BT, 12, C]

    # cls_cost: [BT, S_pred, S_gt]
    cls_gt_exp = cls_gt_flat.clamp(0, C - 1).unsqueeze(1).expand(BT, S_pred, S_gt)
    cls_cost   = -prob.gather(2, cls_gt_exp)                          # [BT, 12, 3]

    # doa_cost: [BT, S_pred, S_gt]
    pred_norm = F.normalize(doa_flat,    dim=-1)                      # [BT, 12, 3-xyz]
    gt_norm   = F.normalize(doa_gt_flat, dim=-1)                      # [BT, 3, 3-xyz]
    doa_cost  = 1.0 - torch.bmm(pred_norm, gt_norm.transpose(1, 2))  # [BT, 12, 3]

    cost_np = (cls_cost + 0.5 * doa_cost).cpu().numpy()   # [BT, 12, 3]
    mask_np = mask_flat.cpu().numpy()                      # [BT, 3]
```

#### `train.py` — Hungarian 루프

cost matrix shape가 `[BT, 12, 3]`으로 바뀌었으므로 `linear_sum_assignment`가
12×(active_count) 행렬을 받는다. 이미 rectangular matrix를 지원하므로 코드 변경 최소:

**변경 전 (line 168-183):**
```python
bt_arr  = np.empty(BT * S, dtype=np.int64)
row_arr = np.empty(BT * S, dtype=np.int64)
col_arr = np.empty(BT * S, dtype=np.int64)
```

**변경 후:**
```python
# 최대 매칭 수 = min(S_pred, S_gt) × BT
max_matches = BT * S_gt
bt_arr  = np.empty(max_matches, dtype=np.int64)
row_arr = np.empty(max_matches, dtype=np.int64)
col_arr = np.empty(max_matches, dtype=np.int64)
```

나머지 루프 로직은 동일 — `linear_sum_assignment`는 rectangular cost를 지원함.

#### `train.py` — confidence target 구성

```python
# 변경 전:
conf_tgt = torch.zeros(BT, S, device=device)

# 변경 후:
conf_tgt = torch.zeros(BT, S_pred, device=device)   # [BT, 12]
# matched된 row_idx만 1.0, 나머지 9개는 0.0
```

---

## 수정 3: Confidence Loss 밸런싱

### 문제
12슬롯 중 최대 3개만 positive. 9:3 = 3:1 불균형으로 confidence head가
"항상 0 출력"에 빠질 수 있음.

### 변경 내용

#### `train.py` — `_compute_single_layer_loss()` 내 presence_loss

**변경 전 (line 206):**
```python
presence_loss = F.binary_cross_entropy_with_logits(conf_flat, conf_tgt)
```

**변경 후:**
```python
# Positive slots에 가중치 부여 (S_pred / max(n_positive, 1))
n_positive = conf_tgt.sum().clamp(min=1)
n_total    = conf_tgt.numel()
pos_weight = torch.tensor(
    [(n_total - n_positive) / n_positive],
    device=device
).clamp(max=10.0)   # 상한 10으로 제한

presence_loss = F.binary_cross_entropy_with_logits(
    conf_flat, conf_tgt,
    pos_weight=pos_weight
)
```

> `pos_weight`는 `[1]` shape 스칼라 텐서. `bce_with_logits`의 `pos_weight` 인자는
> positive target에 곱해지는 가중치로, 여기서는 ~3.0 (9/3)이 됨.

---

## 수정 4: DOA-aware NMS (추론 전용)

### 신규 파일 `nms.py`

```python
"""
SLED v3.2 — DOA-aware Non-Maximum Suppression
===============================================
Over-provisioned slots (12) → NMS → final detections (≤ max_sources).

NMS 조건: 두 슬롯의 DOA cosine similarity > cos_thresh AND 같은 predicted class
→ confidence가 낮은 쪽 제거.
"""

import torch
import torch.nn.functional as F


def doa_nms(
    class_logits: torch.Tensor,
    doa_vecs: torch.Tensor,
    confidences: torch.Tensor,
    cos_thresh: float = 0.9,
    conf_thresh: float = 0.5,
    max_sources: int = 3,
) -> torch.Tensor:
    """
    Parameters
    ----------
    class_logits : [S, C]   raw logits (pre-softmax)
    doa_vecs     : [S, 3]   unit vectors
    confidences  : [S]      after sigmoid
    cos_thresh   : cosine similarity threshold for suppression
    conf_thresh  : minimum confidence to consider a slot active
    max_sources  : maximum number of output detections

    Returns
    -------
    keep_idx : [K]  indices of kept slots, K ≤ max_sources
    """
    # Step 1: confidence threshold
    active_mask = confidences > conf_thresh
    active_idx  = active_mask.nonzero(as_tuple=True)[0]

    if len(active_idx) == 0:
        return active_idx

    # Step 2: sort by confidence descending
    order      = confidences[active_idx].argsort(descending=True)
    active_idx = active_idx[order]

    # Step 3: predicted class for each active slot
    pred_cls = class_logits[active_idx].argmax(dim=-1)   # [K_active]

    # Step 4: greedy NMS
    keep = []
    suppressed = torch.zeros(len(active_idx), dtype=torch.bool,
                             device=doa_vecs.device)

    for i in range(len(active_idx)):
        if suppressed[i]:
            continue

        idx_i = active_idx[i]
        keep.append(idx_i)

        if len(keep) >= max_sources:
            break

        # Suppress remaining slots that have same class AND similar DOA
        remaining = ~suppressed
        remaining[i] = False
        remaining_idx = remaining.nonzero(as_tuple=True)[0]

        if len(remaining_idx) == 0:
            continue

        cos_sim = F.cosine_similarity(
            doa_vecs[idx_i].unsqueeze(0),
            doa_vecs[active_idx[remaining_idx]],
            dim=-1
        )
        same_class = pred_cls[remaining_idx] == pred_cls[i]

        # Suppress: same class AND close DOA
        to_suppress = (cos_sim > cos_thresh) & same_class
        suppressed[remaining_idx[to_suppress]] = True

    if len(keep) == 0:
        return torch.tensor([], dtype=torch.long, device=doa_vecs.device)

    return torch.stack(keep)


def batch_doa_nms(
    class_logits: torch.Tensor,
    doa_vecs: torch.Tensor,
    confidences: torch.Tensor,
    cos_thresh: float = 0.9,
    conf_thresh: float = 0.5,
    max_sources: int = 3,
) -> list[torch.Tensor]:
    """
    Batched NMS for [B, T, S, ...] shaped predictions.

    Parameters
    ----------
    class_logits : [B, T, S, C]
    doa_vecs     : [B, T, S, 3]
    confidences  : [B, T, S]      raw logits (sigmoid applied internally)

    Returns
    -------
    list of [B, T] tensors, each containing kept slot indices for that frame
    """
    B, T, S, C = class_logits.shape
    conf_prob = torch.sigmoid(confidences)   # logits → probability

    results = []
    for b in range(B):
        frame_results = []
        for t in range(T):
            keep = doa_nms(
                class_logits[b, t],   # [S, C]
                doa_vecs[b, t],       # [S, 3]
                conf_prob[b, t],      # [S]
                cos_thresh=cos_thresh,
                conf_thresh=conf_thresh,
                max_sources=max_sources,
            )
            frame_results.append(keep)
        results.append(frame_results)

    return results
```

---

## 수정 5: SLEDv3.forward()에 NMS 경로 추가

### `sled.py`

import 추가:
```python
from .nms import batch_doa_nms
```

`__init__`에 `self.max_sources = max_sources` 저장 (수정 1에서 이미 추가).

#### `forward()` 반환값 변경

**변경 전 (line ~190-210):** matching queries만 heads에 전달.

**변경 후:** 추론 시 전체 12슬롯의 예측을 반환하고, NMS 결과도 함께 포함.

```python
# ── Detection heads for ALL slots (not just n_slots=3) ───────────
layer_preds = []
for layer_out in all_layer_outputs:
    match_out = layer_out[:, :self.n_slots, :].reshape(
        B, T, self.n_slots, -1
    )
    layer_preds.append(self.heads(match_out, B, T))

result = {'layer_preds': layer_preds}

# ── NMS (inference only) ─────────────────────────────────────────
if gt is None and not self.training:
    last_pred = layer_preds[-1]
    nms_indices = batch_doa_nms(
        class_logits = last_pred['class_logits'],
        doa_vecs     = last_pred['doa_vec'],
        confidences  = last_pred['confidence'],
        cos_thresh   = 0.9,
        conf_thresh  = 0.5,
        max_sources  = self.max_sources,
    )
    result['nms_indices'] = nms_indices

# ── DN heads (training only, unchanged) ──────────────────────────
# ... (기존 코드 유지)
```

> **중요:** 학습 시에는 NMS를 사용하지 않음. Hungarian matching이 12→3 매칭을 처리.
> NMS는 순수 추론 경로에서만 동작.

---

## 수정 6: Single→Multi 커리큘럼 학습

### 개념
BAT 논문에서 검증된 전략. Stage I에서 단일 음원 perception을 먼저 학습하고,
Stage II에서 다중 음원을 도입해 slot separation을 학습.

### 변경 내용

#### `train.py` — 학습 루프

**변경 전 (line 563-572):**
```python
for epoch in range(start_epoch, args.epochs + 1):
    # Curriculum: adjust n_dn_groups
    if epoch <= 30:
        model.denoising.n_dn_groups = 5
    else:
        model.denoising.n_dn_groups = 3
```

**변경 후:**
```python
for epoch in range(start_epoch, args.epochs + 1):

    # ── Curriculum: source count + DN groups ──────────────────────
    # Phase 1 (1-40):   단일 음원 위주, DN 5 groups
    # Phase 2 (41-80):  1~2 음원 혼합, DN 3 groups
    # Phase 3 (81-150): 1~3 음원 혼합, DN 3 groups
    # Phase 4 (151+):   다중 음원 비율 높임, DN 3 groups
    if epoch <= 40:
        model.denoising.n_dn_groups = 5
        max_sources_curriculum = 1
    elif epoch <= 80:
        model.denoising.n_dn_groups = 3
        max_sources_curriculum = 2
    elif epoch <= 150:
        model.denoising.n_dn_groups = 3
        max_sources_curriculum = 3
    else:
        model.denoising.n_dn_groups = 3
        max_sources_curriculum = 3   # multi-source 비율 강화 (데이터 로더에서 제어)

    # DataLoader에 현재 phase의 max_sources 전달
    if hasattr(train_loader.dataset, 'set_max_sources'):
        train_loader.dataset.set_max_sources(max_sources_curriculum)
```

#### 데이터셋 클래스에 커리큘럼 인터페이스 추가

> 이 부분은 `sled/dataset/torch_dataset.py`에 구현해야 함.
> 아래는 인터페이스 명세로, 데이터셋 클래스에 다음 메서드를 추가:

```python
def set_max_sources(self, max_sources: int):
    """커리큘럼 학습용: 이 epoch에서 사용할 최대 음원 수 제한.

    max_sources=1이면 단일 음원 샘플만 반환.
    max_sources=2이면 1~2 음원 샘플 반환.
    max_sources=3이면 제한 없음 (원본 데이터셋 그대로).

    구현 방법:
      - __getitem__에서 활성 음원 수가 max_sources를 초과하면
        랜덤으로 초과분의 mask를 False로 설정
      - 또는 미리 인덱스를 필터링해 해당 조건의 샘플만 선택
    """
    self._max_sources = max_sources
```

---

## 수정 7: Moderate Scaling

### 개념
구조 수정 + NMS 적용 후에도 성능 부족 시 moderate scaling 적용.
5배가 아니라 **2~3배** 정도가 비용 대비 효율적.

### 변경 내용

#### `sled.py` — 기본값 변경

```python
# 변경 전:
def __init__(self, sofa_path: str, d_model: int = 256,
             n_slots: int = 12, n_classes: int = 209,
             n_decoder_layers: int = 4, n_conformer_layers: int = 4, ...):

# 변경 후 (scaling 적용 시):
def __init__(self, sofa_path: str, d_model: int = 384,
             n_slots: int = 12, n_classes: int = 209,
             n_decoder_layers: int = 6, n_conformer_layers: int = 8, ...):
```

#### `train.py` — argparse 기본값

```python
# 변경 전:
parser.add_argument('--d-model', type=int, default=256)

# 변경 후:
parser.add_argument('--d-model', type=int, default=384)
```

#### encoder GroupNorm 호환성 확인

d_model=384일 때 GroupNorm의 num_groups가 나눠떨어지는지 확인 필요:

```python
# encoder.py의 GroupNorm들:
nn.GroupNorm(8, out_ch)    # out_ch=64,128,256 → 384도 8로 나눠짐 ✓
nn.GroupNorm(32, d_model)  # 384 / 32 = 12 ✓
# decoder.py 등도 동일하게 확인
```

384는 8, 16, 32 모두로 나눠지므로 GroupNorm 호환 문제 없음.

### 예상 파라미터 변화

| 설정 | 파라미터 수 (approx) |
|------|---------------------|
| 현재 (d=256, conf=4, dec=4, slots=3) | ~12M |
| v3.2 slots만 (d=256, slots=12) | ~13M |
| v3.2 + moderate scaling (d=384, conf=8, dec=6, slots=12) | ~35M |

---

## 데이터 증강 권장 사항

코드 수정은 아니지만 성능에 큰 영향을 미치므로 기록:

### pyroomacoustics 기반 데이터 합성 확장

```
현재 데이터셋의 다중 음원 비율이 부족하면 아래 조건으로 합성 데이터 추가:
- 방 크기: small(3-5m), medium(8-12m), large(20-30m)
- RT60: 0.2s ~ 1.2s (6단계)
- 음원 수: 1, 2, 3개 균등 분포 (최소 각각 30% 이상)
- 음원 간 각도 차이: 10° ~ 180° (균등 분포)
  → 특히 30° 이하 근접 음원 케이스를 전체의 20% 이상 포함
- SNR: -5dB ~ 20dB
```

### 데이터 밸런싱

```
다중 음원 학습 효과를 극대화하려면:
- 1음원 : 2음원 : 3음원 = 2 : 4 : 4 비율 (다중 음원 과대표집)
- 근접 음원(< 30° 차이) 케이스를 별도로 hard example로 추가
```

---

## 적용 순서 (권장)

```
1. 수정 1-3: slots 12 + Hungarian 비대칭 + confidence 밸런싱 (한 번에 적용)
2. 수정 4-5: NMS 구현 + forward에 통합
3. 학습하여 baseline 성능 확인
4. 수정 6: 커리큘럼 학습 적용하여 재학습
5. 성능 부족 시 수정 7: moderate scaling 적용
6. 데이터 증강은 1-5와 병렬로 진행 가능
```

---

## 체크리스트

- [ ] `sled.py`: `n_slots=12`, `max_sources=3` 파라미터 추가
- [ ] `decoder.py`: slot_queries [12, d] + spatial prior 30° 간격
- [ ] `train.py`: cost matrix를 `[BT, 12, 3]`으로 구성
- [ ] `train.py`: Hungarian 매칭 배열 크기를 `BT * S_gt`로 조정
- [ ] `train.py`: confidence target을 `[BT, 12]`로 확장
- [ ] `train.py`: `pos_weight` 적용하여 confidence loss 밸런싱
- [ ] `nms.py`: 신규 파일 생성 (DOA cosine + class 조건부 NMS)
- [ ] `sled.py`: 추론 경로에 `batch_doa_nms` 호출 추가
- [ ] `train.py`: 커리큘럼 phase 4단계 구현 (epoch → max_sources 매핑)
- [ ] 데이터셋 클래스에 `set_max_sources()` 인터페이스 추가
- [ ] (성능 부족 시) d_model 256→384, conformer 4→8, decoder 4→6
- [ ] (성능 부족 시) GroupNorm num_groups 호환성 확인
