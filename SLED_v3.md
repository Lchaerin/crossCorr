# SLED v3 — Binaural Sound Localization & Event Detection

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [DINO와의 비교 — 설계 철학](#2-dino와의-비교--설계-철학)
   - [2.4 DINO에서 직접 차용한 요소 — 코드 레벨 대응](#24-dino에서-직접-차용한-요소--코드-레벨-대응)
3. [전체 아키텍처](#3-전체-아키텍처)
4. [좌표계 규약](#4-좌표계-규약)
5. [데이터셋](#5-데이터셋)
   - [5.6 MRS Mix 데이터셋](#56-mrs-mix-데이터셋-data_mrs_mix)
   - [5.7 MRS Balanced 데이터셋](#57-mrs-balanced-데이터셋-data_mrs_balanced)
6. [오디오 전처리기 (AudioPreprocessor)](#6-오디오-전처리기-audiopreprocessor)
7. [인코더 (SLEDEncoder)](#7-인코더-sledencoder)
8. [디코더](#8-디코더)
9. [Detection Heads](#9-detection-heads)
10. [Loss 함수](#10-loss-함수)
11. [학습](#11-학습)
12. [추론 및 시각화](#12-추론-및-시각화)
    - [12.4 visualize.py 주요 수정 이력](#124-visualizepy-주요-수정-이력)
13. [Ablation Study](#13-ablation-study)
    - [13.5 MRS 체크포인트 비교](#135-mrs-체크포인트-비교-스크립트)
14. [파일 구조](#14-파일-구조)
15. [주요 하이퍼파라미터](#15-주요-하이퍼파라미터)

---

## 1. 프로젝트 개요

**SLED (Sound Localization and Event Detection) v3**는 양이(binaural) 스테레오 오디오를 입력으로 받아 매 프레임마다 동시에 존재하는 음원들의 **방위각/고도각(DOA)** 과 **음원 종류(class)** 를 탐지하는 end-to-end 딥러닝 모델이다.

### 핵심 특징

- **입력**: 48 kHz 스테레오 WAV (`[B, 2, N]`)
- **출력**: 프레임마다 최대 3개 음원의 class · DOA · presence
- **HRTF 활용**: SOFA 형식의 Head-Related Transfer Function으로 공간 단서 추출
- **End-to-end**: 파형 → 탐지 결과까지 단일 순전파
- **인과적(causal)**: 미래 프레임을 보지 않아 실시간 처리 가능한 구조

---

## 2. DINO와의 비교 — 설계 철학

SLED v3는 이미지 객체 탐지 모델인 **DINO**(DETR with Improved deNoising anchOr boxes, Zhang et al. 2022)의 핵심 아이디어를 음원 탐지 도메인으로 이식한 구조다. 두 모델은 목적 함수와 디코더 학습 전략을 공유하지만, 도메인 차이에서 비롯된 중요한 설계 차이가 있다.

### 2.1 공통 설계 원칙

| 항목 | DINO | SLED v3 |
|---|---|---|
| **쿼리 기반 탐지** | 고정 수의 학습가능 쿼리 슬롯 | 고정 수(3)의 슬롯 쿼리 |
| **이분 매칭 (Hungarian)** | 예측 박스 ↔ GT 박스 최적 할당 | 예측 DOA/class ↔ GT 음원 최적 할당 |
| **반복적 정밀화** | 레이어별 박스 좌표 delta 예측 | 레이어별 DOA 단위벡터 피드백 |
| **Contrastive DeNoising** | GT 박스에 노이즈 주입 → 복원 학습 | GT DOA에 구면 노이즈 주입 → 복원 학습 |
| **분리된 존재 확인** | "no-object" 클래스 OR conf 헤드 | 전용 presence 헤드 (BCE) |
| **Focal Loss** | 클래스 분류에 사용 | 클래스 분류에 사용 |

### 2.2 주요 차이점

#### ① 탐지 대상: 박스 → DOA 단위벡터

DINO는 이미지 좌표계의 2D 경계 박스 `(cx, cy, w, h)` 를 예측한다. 좌표가 유클리드 공간에 있으므로 L1 + GIoU 손실로 학습한다.

SLED는 3D 단위구 위의 방향 벡터 `(x, y, z)` 를 예측한다. 단위구는 유클리드 공간이 아니므로 코사인 유사도 기반 손실을 사용하고, 각 레이어에서 예측된 raw vector를 **L2 정규화**하여 단위벡터를 유지한다.

```
# DINO: 박스 좌표 delta 예측
box_pred = sigmoid(box_raw + delta)   # [0,1]^4로 클램핑

# SLED: DOA 단위벡터 정밀화
doa_pred = normalize(doa_raw)         # 단위구로 투영
```

#### ② 쿼리 초기화: 앵커 기반 → Cross-Attention 기반

DINO는 두 종류의 쿼리를 분리한다:
- **Content query**: 학습 가능한 파라미터 (무엇을 찾을지)
- **Positional query**: 인코더가 제안한 상위 K개 앵커 박스 (어디를 찾을지)

SLED는 이 두 역할을 **CrossAttentionQuerySelector** 하나로 통합한다. 학습 가능한 슬롯 파라미터(Q)가 7개의 multi-scale 오디오 특징(KV)에 cross-attention하여 초기 쿼리를 생성한다. "공간 앵커" 개념 대신, 다주파수 오디오 맥락으로부터 슬롯 내용과 위치 모두를 동시에 초기화한다.

```
# DINO: 분리된 content + positional query
content_q  = learnable_params                   # 학습 가능
pos_q      = top_k_encoder_proposals            # 인코더 제안

# SLED: 통합 cross-attention 초기화
slot_q = CrossAttention(Q=slot_params,          # 학습 가능
                        KV=7_multiscale_feats)  # 완전 미분 가능
```

#### ③ 기억(Memory): 전체 피처맵 → 7-토큰 압축 메모리

DINO의 디코더는 **인코더 전체 출력** (이미지의 경우 수백~수천 토큰)에 cross-attention한다. 이미지는 임의 길이가 없으므로 이 토큰 수가 고정된다.

SLED의 디코더는 오디오의 각 시간 프레임을 독립적으로 처리해야 한다 (`B×T`개의 인스턴스). 전체 피처맵을 메모리로 사용하면 `B×T × token_count`로 메모리가 폭발한다. 이를 해결하기 위해 multi-scale 특징을 **7개 압축 토큰**으로 정리한다:

```
7 tokens = [P3_lo, P3_hi, P4_lo, P4_hi, P5_lo, P5_hi, enc_out]
         ↑ 저주파 음색    ↑ 고주파 음색   ↑ 전역 시간 맥락
```

이 7개 토큰은 주파수별로 분리된 오디오 표현을 제공하며, 디코더가 음원 유형에 따라 선택적으로 특정 주파수 대역을 참조할 수 있게 한다.

#### ④ Contrastive DeNoising: 박스 노이즈 → 구면 노이즈

DINO의 DN은 GT 박스 좌표에 균일 분포 노이즈를 더해 오염시킨다.

SLED의 DN은 GT DOA 단위벡터를 **구면 상에서 오염**시켜야 한다. 유클리드 가산 노이즈를 적용하면 단위구 밖으로 벗어나므로, 3D 가우시안 노이즈를 더한 뒤 L2 정규화하여 구면 위의 점을 유지한다:

```python
# Positive DN (작은 노이즈, scale=0.2)
noise_pos = gt_doa + randn() * 0.2
doa_dn_pos = normalize(noise_pos)

# Negative DN (큰 노이즈, scale=0.8)
noise_neg = gt_doa + randn() * 0.8
doa_dn_neg = normalize(noise_neg)
```

Positive DN 쿼리는 원래 GT class + 소교란 DOA로 초기화되고, Negative DN 쿼리는 무작위 셔플 class + 대교란 DOA로 초기화된다. 디코더는 positive는 올바르게 복원하고 negative는 비활성(confidence=0)으로 억제하도록 학습된다.

#### ⑤ 음원 부재 처리: no-object 클래스 → 전용 Presence 헤드

DINO(DETR 계열)는 "no object"를 `n_classes+1`번째 클래스로 취급하여 분류 헤드에 포함한다. 이는 분류와 탐지를 하나의 소프트맥스로 묶어 상호 간섭이 생길 수 있다.

SLED는 이 둘을 **완전히 분리**한다:
- `class_head` (209-way logits): 음원이 있을 때 무엇인지만 담당
- `conf_head` (binary logit): 음원이 있는지 없는지만 담당

이 분리 덕분에 empty 클래스로 인한 클래스 불균형 문제가 없고, presence와 class를 독립적으로 조정할 수 있다.

#### ⑥ 인과성 (Causality)

DINO는 이미지 전체를 한 번에 보는 구조로, 인과성이 필요 없다.

SLED는 실시간 스트리밍을 목표로 **완전한 인과 구조**를 유지한다:
- `CausalConv2d`: 시간 방향 왼쪽 패딩만 적용
- `CausalConformerBlock`: 상삼각 self-attention mask로 미래 프레임 차단
- `CausalBiFPN`: 상향/하향 융합 모두 미래 참조 없음

### 2.3 구조 대응 요약

```
DINO                            SLED v3
─────────────────────────────   ──────────────────────────────────
ResNet/Swin Backbone            CausalConv Stem + CausalBiFPN
                                (causal multi-scale extraction)

Multi-scale Feature Maps        7-token Compressed Memory
(HW tokens per scale)           [P3_lo/hi, P4_lo/hi, P5_lo/hi, enc]

Deformable DETR Encoder         CausalConformerBlock × 4
(cross-scale attention)         (temporal modeling, causal)

Top-K Anchor Proposal           CrossAttentionQuerySelector
(encoder → positional query)    (slot params cross-attend to 7 tokens)

Box Coordinates (cx,cy,w,h)     DOA Unit Vector (x,y,z) on S²
L1 + GIoU Loss                  Cosine Distance Loss

Iterative Box Refinement        Iterative DOA Refinement
  delta(box) per layer            normalize(doa_raw) per layer
  box coord feedback              doa positional embedding feedback

Contrastive DeNoising           Contrastive DeNoising
  Euclidean box noise              Spherical DOA noise + class shuffle
  no-object class target           confidence=0 target for negative DN

no-object (N+1 class)           Separate Presence Head (BCE)
```

### 2.4 DINO에서 직접 차용한 요소 — 코드 레벨 대응

아래 표는 DINO 논문/구현의 어떤 아이디어가 SLED의 어떤 파일·클래스에 직접 반영되었는지 정리한다.

| DINO 원출처 | 차용 내용 | SLED 구현 위치 |
|---|---|---|
| **Hungarian bipartite matching** (DETR 원조) | 예측-GT 쌍을 비용행렬로 최적 할당. 비용 = class focal + DOA cosine distance + loudness | `sled/train.py` → `hungarian_match()` |
| **Auxiliary decoder losses** (DETR 원조) | 모든 디코더 레이어 출력에 동일한 loss 적용 → gradient가 초기 레이어까지 풍부하게 전달 | `sled/train.py` → `compute_losses()` (레이어별 루프) |
| **Contrastive DeNoising — positive group** (DINO §3.1) | GT 라벨 + 소교란(scale=0.2) 쿼리 → 원래 GT 복원 학습. 학습 안정성·수렴 속도 향상 | `sled/model/decoder.py` → `ContrastiveDeNoising` |
| **Contrastive DeNoising — negative group** (DINO §3.1) | 셔플 클래스 + 대교란(scale=0.8) 쿼리 → confidence=0 억제 학습. no-object 판별 능력 강화 | `sled/model/decoder.py` → `ContrastiveDeNoising` |
| **DN 격리 self-attention mask** (DINO §3.1) | matching 쿼리 ↔ DN 쿼리 간 attention을 완전 차단하여 두 그룹이 서로 영향받지 않도록 함 | `sled/model/sled.py` → `_build_dn_mask()` |
| **Iterative refinement per decoder layer** (DINO §3.2) | 각 레이어가 이전 레이어 예측을 positional embedding으로 받아 점진적으로 정밀화 | `sled/model/decoder.py` → `IterativeRefinementDecoder` |
| **Multi-scale feature pyramid** (Deformable DETR) | 백본의 여러 스케일 특징을 별도로 유지해 small/large 객체 모두 탐지 | `sled/model/encoder.py` → `CausalBiFPN` (P3/P4/P5) |
| **Focal Loss for classification** (RetinaNet → DETR) | 클래스 불균형 완화. γ=2, α=0.25 | `sled/train.py` → `focal_loss()` |

#### 차용 시 변형 요약

| 원본 DINO 알고리즘 | SLED에서 바뀐 점 | 이유 |
|---|---|---|
| Box noise: `box + U(−λ, λ)` | DOA noise: `normalize(doa + N(0, σ²)·I)` | 탐지 대상이 유클리드 좌표가 아닌 단위구(S²) 위의 점 |
| Anchor positional query (top-K 인코더 제안) | CrossAttentionQuerySelector (슬롯 파라미터 → 7 토큰 cross-attention) | 오디오는 공간 앵커 개념이 없음; 주파수 맥락에서 슬롯 초기화 |
| Full feature map as decoder memory | 7-token compressed memory per frame | `B×T` 인스턴스를 병렬 처리할 때 메모리 폭발 방지 |
| no-object as (N+1)-th class | 분리된 `conf_head` (binary BCE) | class head와 presence head 간섭 제거 |
| Bidirectional self-attention | 상삼각 인과 마스크 (causal self-attention) | 실시간 스트리밍 요건 |

---

## 3. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│  입력: 스테레오 파형  [B, 2, N]                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ AudioPrep.  │  STFT → mel / ILD / IPD / HRTF
                    └──────┬──────┘
                     ↙           ↘
          [B, C, 64, T]      [B, 64, 32]   ← HRTF 히트맵 (C는 활성 채널 수)
                     │
            ┌────────▼────────┐
            │  SLEDEncoder    │
            │  CausalConvStem │  P3/P4/P5  (GroupNorm)
            │  SEBlock        │
            │  CausalBiFPN×2  │
            │  CausalConformer×4
            │  + HRTF bias    │  ← [B,d] broadcast to [B,T,d]
            └────────┬────────┘
          multi_scale│(7 tokens) + enc_out [B,T,d]
                     │
            ┌────────▼──────────────┐
            │CrossAttentionQuery    │  slot params(Q) → 7 tokens(KV)
            │Selector               │  (fully differentiable)
            └────────┬──────────────┘
                     │  queries [B*T, 3, d]   ← matching queries
                     │  memory  [B*T, 7, d]   ← 7 multi-scale tokens
                     │  (학습 시: DN 쿼리 concat + self-attn mask)
            ┌────────▼────────────────┐
            │ IterativeRefinement     │  self-attn + cross-attn(7)
            │ Decoder × 4            │  + DOA positional feedback/layer
            └────────┬────────────────┘
                     │  4개 레이어 각각 출력 저장
            ┌────────▼────────┐
            │ DetectionHeads  │  class / DOA / loudness / presence
            └────────┬────────┘
                     │
          ┌──────────▼──────────┐
          │ 출력: [B, T, 3, *]  │  class_logits / doa_vec / confidence
          └─────────────────────┘
```

---

## 4. 좌표계 규약

### MRSAudio 좌표계 규약

MRS npy 파일은 `(right_rel, forward_rel, up_rel, time_ms)` 형태의 카테시안 좌표로 저장된다.
이를 방위각으로 표현하면: **az = atan2(forward, right)**

| 방향 | (right, fwd) | atan2(fwd, right) |
|---|---|---|
| 정면 | (0, +1) | **90°** |
| 오른쪽 | (+1, 0) | 0° |
| **왼쪽** | (-1, 0) | **180°** |
| 뒤 | (0, -1) | -90° (=270°) |

→ **반시계방향(CCW), 90°=정면, 180°=왼쪽**

SLED 규약(CW, 0°=정면)으로 변환:
```python
doa_sled = normalize([forward, right, up])   # x=fwd, y=right, z=up
# SLED azimuth = atan2(right, forward) → 0°=정면, 90°=오른쪽 (CW)
```
이 변환으로 MRS CCW → SLED CW 가 올바르게 처리된다.

---

### SOFA vs SLED 방위각

SOFA: **반시계(CCW)**, 0°=앞, 90°=왼쪽
SLED: **시계(CW)**, 0°=앞, 90°=오른쪽

```
az_sled = (-az_sofa) % 360
```

### 3D DOA 단위벡터 (SLED 규약)

```
x = cos(el) · cos(az_sled)   ← 앞/뒤
y = cos(el) · sin(az_sled)   ← 오른쪽이 양수
z = sin(el)                  ← 위가 양수
```

---

## 5. 데이터셋

### 5.1 음원 파일

- 기본 경로: `soud_effects/` (FSD50K 심볼릭 링크 레이아웃)
- 클래스 디렉터리 방식: `soud_effects/Dog/12345.wav`
- 클래스 수: **209개** (FSD50K 199 카테고리 + 합성 톤 10개)
- `class_map.json`: `{"카테고리/파일명": class_id}` 형식

### 5.2 씬 합성 파이프라인 (`synthesizer.py`)

한 씬 = 30초 스테레오 WAV + 밀집 어노테이션

```
1. schedule_events()
   - 3~8개 음원, 동시 활성 ≤ 3
   - 각 2~12초 지속, 페이드 50ms
   - start_sample / end_sample (48 kHz 기준) 저장

2. mix_binaural()
   - 음원 × HRTF 임펄스 응답 FFT 컨볼루션
   - 피크 정규화 + 백색 잡음 (SNR 15~35 dB)

3. compute_dense_annotations()
   - 20ms 프레임(960 샘플) 단위 슬롯별 GT
   - 겹침 판정: start_sample < frame_end AND end_sample > frame_start
   - 출력: cls[T,3], doa[T,3,3], loud[T,3], mask[T,3]
```

### 5.3 데이터 분할

| 분할 | 씬 수 | base_id |
|---|---|---|
| train | 10,000 | 0 |
| val | 1,000 | 10,000 |
| test | 500 | 11,000 |

### 5.4 데이터 레이아웃

```
data/
  meta/
    class_map.json    {sfx_key: class_id}  — 209 클래스
    split.json
  audio/{train,val,test}/   scene_XXXXXX.wav + .json
  annotations/{train,val,test}/
    scene_XXXXXX_cls.npy   [T, 3]    int16  (-1=비활성)
    scene_XXXXXX_doa.npy   [T, 3, 3] float16
    scene_XXXXXX_loud.npy  [T, 3]    float16 (dBFS)
    scene_XXXXXX_mask.npy  [T, 3]    bool
```

### 5.5 SLEDDataset

- 씬에서 무작위 `window_frames` 구간 샘플링 → 256 × 20ms = **5.12초**
- **Stereo Channel Swap**: L↔R 교환 + DOA y 성분 부호 반전 (50%)
- **음량 임계값 필터링** (`min_loudness_db=-60.0` dBFS):
  - RMS < 임계값이면 `mask=False`, `cls=-1` 처리
  - fade-in/out 꼬리, 사실상 무음 구간에서 혼동 방지

---

### 5.6 MRS Mix 데이터셋 (`data_mrs_mix/`)

`build_mrs_mix_dataset.py`가 생성하는 **MRSAudio 기반 다중 음원 합성 데이터셋**.
기존 합성 데이터셋(`data/`)과 **완전히 동일한 포맷** — 같은 `SLEDDataset` / `train.py`로 바로 사용 가능.

#### 생성 배경

MRS 원본 녹음은 WAV당 음원이 항상 1개(12,057개 구간 전수 확인). 다중 음원 탐지 학습을 위해 여러 MRS 세그먼트를 가산 합성하여 최대 3개 동시 음원 씬을 만든다.

#### 주요 설계

| 항목 | 값 |
|---|---|
| 씬 길이 | 30초 (T=1500 프레임) |
| 최대 동시 음원 | 3개 |
| 소스 수 분포 | 단일 20% / 2중 40% / 3중 40% |
| 소스 정규화 | TARGET\_RMS=0.05 → ±6 dB 랜덤 게인 |
| 클래스 매핑 | MRS 이벤트명 → FSD50K class ID (27종) |
| 클리핑 방지 | peak > 0.95 시 전체 스케일 다운 + loud\_arr 보정 |

#### 음원 풀 분할

```
train pool: MRSSound/sound001 ~ sound166  (80%, 9,613 세그먼트)
test  pool: MRSSound/sound167 ~ sound208  (20%, 2,386 세그먼트, held-out)
val은 train pool에서 다른 시드로 조합
```

#### DOA 변환

MRS npy `[T_npy, 4]` (right, fwd, up, time_ms) → SLED 단위벡터:
```python
doa_raw = [forward, right, up]   # MRS → SLED 좌표 순서
doa_unit = normalize(doa_raw)    # 단위구 투영
```
프레임 간격 50ms → 20ms로 선형 보간.

#### 어노테이션 특이사항

JSON의 `azimuth/elevation`은 `null` (DOA가 프레임마다 변함).
`visualize.py`의 `--gt-json` 옵션 사용 시 자동으로 npy 파일에서 GT를 읽음.

#### 생성 명령어

```bash
python build_mrs_mix_dataset.py \
    --mrs-root ./MRSAudio/MRSLife/MRSSound \
    --out-dir  ./data_mrs_mix \
    --n-train  6000 \
    --n-val    750  \
    --n-test   250  \
    --seed     42
# 속도: ~30 씬/초
```

#### 학습 명령어

```bash
# Scratch
python -m sled.train \
    --dataset-root  ./data_mrs_mix \
    --n-classes     209 \
    --epochs        200 \
    --batch-size    64 \
    --lr            3e-4 \
    --window-frames 48

# Finetune (pretrained 체크포인트에서 시작)
python -m sled.train \
    --dataset-root  ./data_mrs_mix \
    --resume        ./checkpoints/sled_best.pt \
    --epochs        240 \        # 기존 epoch + 50
    --lr            3e-5 \
    --weight-decay  1e-3
```

#### 클래스 공간

`data_mrs_mix`와 `data_custom_hrtf`(합성 데이터)는 **동일한 FSD50K 209-class 공간**을 공유한다.
MRS mix는 그 중 15개 클래스만 실제로 사용(IDs: 11,26,34,36,39,48,57,87,111,133,137,160,165,177,194).
class ID 충돌 없이 두 데이터셋을 바로 혼합 학습 가능.

#### 주의: MRS Mix의 낮은 Loss ≠ 높은 성능

MRS mix는 실제 클래스가 15개뿐, DOA 분포가 편중, 음원 조합이 반복적이라
합성 데이터보다 Loss가 빨리 떨어지지만 실제 테스트 성능은 낮을 수 있음.
합성 데이터 pretrained → MRS mix finetune 순서가 가장 효과적.

#### DOA 편향 분석

`data_mrs_mix` train 어노테이션의 active 프레임 azimuth 분포 (SLED CW, 0°=정면):

```
  -60°~ 60° (전방):  ~163,000 프레임   ← 집중
  90°~180° / -90°~-180° (후방):  ~18,000 프레임
  전방:후방 비율 ≈ 9:1
```

원인: MRS 실험 셋업이 리스너 앞쪽 위주. 이 편향으로 모델이 후방 음원도 전방으로 오인하는 문제 발생.
→ 해결책: `data_mrs_balanced` (§5.7) 또는 합성 데이터와 혼합 학습.

---

### 5.7 MRS Balanced 데이터셋 (`data_mrs_balanced/`)

`build_mrs_balanced_dataset.py`가 생성하는 **azimuth 균등 분포** MRS 데이터셋.
포맷은 `data_mrs_mix`와 완전히 동일 — 같은 `SLEDDataset` / `train.py`로 바로 사용 가능.

#### 균형화 방법

전체 세그먼트를 azimuth bin별로 분류하고 `weight = 1 / bin_count` 부여:

```
bin_i 확률 = (bin_count_i × weight_i) / Σ(bin_count_j × weight_j)
           = 1 / N_bins   (모든 bin 동일)
```

씬 생성 시 이 가중치로 세그먼트를 샘플링 → 모든 12개 방향 bin이 동일 확률.

#### Train pool azimuth 분포 (sound001~166, 12 bins)

```
  -180~-150°:    86 segs  → weight 1/86   (후방, 희귀)
  -150~-120°:    95 segs  → weight 1/95
  -120~ -90°:   436 segs
   -90~ -60°: 1,217 segs
   -60~ -30°: 2,140 segs
   -30~   0°: 2,335 segs
     0~  30°: 2,063 segs
    30~  60°: 1,914 segs
    60~  90°: 1,060 segs
    90~ 120°:   488 segs
   120~ 150°:   121 segs
   150~ 180°:    81 segs  → weight 1/81   (후방, 희귀)
```

희귀 후방 세그먼트(81~95개)는 같은 클립이 반복 등장하지만,
랜덤 시작 위치 + ±6 dB 게인 증강으로 다양성 확보.

#### 생성 명령어

```bash
python build_mrs_balanced_dataset.py \
    --mrs-root ./MRSAudio/MRSLife/MRSSound \
    --out-dir  ./data_mrs_balanced \
    --n-train  6000 \
    --n-val    750  \
    --n-test   250  \
    --n-bins   12   \
    --seed     42
```

---

## 6. 오디오 전처리기 (AudioPreprocessor)

`sled/model/preprocessor.py`

**입력**: `[B, 2, N]` 스테레오 파형
**출력**: `([B, C, 64, T], ch5|None)` — C는 활성 채널 수

### 6.1 STFT 파라미터

| 파라미터 | 값 |
|---|---|
| 샘플레이트 | 48,000 Hz |
| n_fft | 2,048 (42.7 ms) |
| hop_length | 960 (20 ms) |
| 윈도우 | Hann |
| n_mels | 64 |
| fmin / fmax | 20 / 16,000 Hz |

### 6.2 채널 구성

| 채널 | 이름 | 계산 | 단서 유형 |
|---|---|---|---|
| 0 | L-mel | `10·log10(mel_fb @ pow_L + ε)` | 스펙트럼 (분류) |
| 1 | R-mel | `10·log10(mel_fb @ pow_R + ε)` | 스펙트럼 (분류) |
| 2 | **ILD** | `10·log10(mel_L / mel_R + ε)` | 수평 위치 (ITD 저주파 보완) |
| 3 | **sin(IPD)** | 정규화 CSD 허수부 | 도달 시간 차 (저주파 DOA) |
| 4 | **cos(IPD)** | 정규화 CSD 실수부 | 도달 시간 차 (저주파 DOA) |

활성 채널 수 `C = 2 + use_ild + 2×use_ipd` (ablation 옵션에 따라 2~5).

**CSD (Cross-Spectral Density)**:
```
csd[B,F,T] = X_L * conj(X_R)
mel_csd     = mel_fb @ csd      → [B, 64, T] complex (mel 대역 평균)
norm        = |mel_csd| + ε
sin(IPD)    = mel_csd.imag / norm
cos(IPD)    = mel_csd.real / norm
```

### 6.3 채널 5 — HRTF 교차상관 히트맵

음원 방향 추정을 위한 **방위각 × 고도각 2D 공간 히트맵** (고정 크기 `[B, 64, 32]`).

**사전 계산** (초기화 시, SOFA 파일 기반):
```
W_real[M, F] = Re(HRTF_R * conj(HRTF_L))
W_imag[M, F] = Im(HRTF_R * conj(HRTF_L))
norm_hr_sq, norm_hl_sq[M, F]
az_bin_idx[M]   ← 방위각 → 64 bin 양자화
elevations[M]   ← 고도각 (도)
```

**순전파** (윈도우당 1회):
```
1. STFT 구간에서 8 프레임 균등 샘플 → 평균 CSD [B, F]

2. M개 방향 교차상관:
   corr[B, M] = (W_real @ csd_r - W_imag @ csd_i)
                / sqrt(norm_hr_sq @ pow_avg × norm_hl_sq @ pow_avg + ε)

3. 2D scatter: corr → [B, 64_az, 32_el]
   az: 5.625°/bin (0°~360°), el: 5.625°/bin (-90°~+90°)
```

이 히트맵은 HRTF와 입력 신호의 공간적 일치도를 나타내며, 단일 모델 forward 내에서 물리 기반 공간 단서를 명시적으로 제공한다.

---

## 7. 인코더 (SLEDEncoder)

`sled/model/encoder.py`

**입력**: `[B, C, 64, T]` + `[B, 64, 32]` (HRTF ch5, optional)
**출력**: `multi_scale_feats` (7개 × `[B, T, d]`), `enc_out [B, T, d]`

### 7.1 Causal Conv Stem

```
입력 [B, C, 64, T]
  → stem3: CausalConv2d(C→64,  freq_stride=2) + GroupNorm + GELU → P3 [B,  64, 32, T]
  → stem4: CausalConv2d(64→128, freq_stride=2)                   → P4 [B, 128, 16, T]
  → stem5: CausalConv2d(128→256,freq_stride=2)                   → P5 [B, 256,  8, T]
  → SEBlock(256)                                                  → P5 (채널 재보정)
  → proj3/4/5: Conv2d(→d_model, 1×1)                             → 모두 d_model 채널
```

**CausalConv2d**: 시간 방향으로 `(kernel_size−1)` 왼쪽 패딩 → 미래 참조 없음
**GroupNorm**: BatchNorm 대신 사용 — 소배치·스트리밍 환경에서 안정적

### 7.2 SEBlock (Squeeze-and-Excitation)

```
P5 → global avg pool (F,T) → [B, 256]
   → Linear(256→16) → ReLU → Linear(16→256) → Sigmoid
   → scale × P5
```

채널별 중요도 학습 (주파수 대역 선택).

### 7.3 CausalBiFPN × 2

3 스케일 양방향 피처 융합 (fast-normalised weighted sum):

```
Top-down:  P5 ─upsample→ fuse P4 → P4_td
           P4_td ─upsample→ fuse P3 → P3_td

Bottom-up: P3_td ─pool→ fuse P4_td → P4_bu
           P4_bu ─pool→ fuse P5    → P5_bu

fused = Σ (w_i / Σw_j · feat_i)   (학습 가능한 양수 가중치)
```

모든 정규화: **GroupNorm(8, d_model)**

### 7.4 7-Token Multi-scale Features

BiFPN 이후 각 스케일을 주파수 상/하단으로 분리 → 6개 + Conformer 출력 1개:

```
P3[B,d,32,T]: lo=lower_half_avg [B,T,d], hi=upper_half_avg [B,T,d]  → 2개
P4[B,d,16,T]: lo, hi                                                  → 2개
P5[B,d, 8,T]: lo, hi                                                  → 2개
enc_out[B,T,d]   (Conformer 이후)                                     → 1개
합계: 7개 × [B,T,d]
```

이 7개 토큰이 CrossAttentionQuerySelector의 후보(KV)이자 디코더의 메모리(KV)가 된다.

### 7.5 Temporal Encoder — CausalConformer × 4

```
P5 [B, 256, 8, T]
  → reshape [B, d×8, T]
  → Conv1d(d×8→d, k=1) + GroupNorm(32) + GELU
  → permute [B, T, d]
  → CausalConformerBlock × 4

CausalConformerBlock:
  x → FF1 (half residual, SiLU)
    → Causal MHSA (상삼각 mask)
    → Causal Depthwise Conv (왼쪽 패딩)
    → FF2 (half residual)
    → LayerNorm
```

### 7.6 HRTF Projection

ch5 `[B, 64, 32]`를 전역 공간 임베딩으로 변환 후 enc_out에 broadcast 합산:

```
ch5 [B, 64, 32]
  → Flatten [B, 2048]
  → Linear(2048→d) + GELU + Linear(d→d) + LayerNorm  → [B, d]
  → unsqueeze(1) broadcast add → enc_out [B, T, d]
```

`use_hrtf_corr=False`이면 이 모듈 자체가 생성되지 않는다 (dead weight 없음).

---

## 8. 디코더

`sled/model/decoder.py`

### 8.1 CrossAttentionQuerySelector

DINO의 top-K 앵커 제안과 달리, 학습 가능한 슬롯 파라미터가 7개 multi-scale 토큰에 직접 cross-attention하여 초기 쿼리를 생성한다:

```
candidates [B*T, 7, d]   ← 7 multi-scale tokens (KV)
slot_q     [B*T, 3, d]   ← learnable slot params (Q)

output = CrossAttention(Q=slot_q, KV=candidates)  → [B*T, 3, d]
```

- topk·detach 없음 → **완전 미분 가능**
- 각 슬롯은 학습을 통해 특정 주파수 대역 또는 공간 영역에 특화될 수 있음

### 8.2 ContrastiveDeNoising (학습 시만)

DINO의 CDN을 구면 DOA 공간으로 이식:

| 종류 | DOA 노이즈 | Class | 학습 목표 |
|---|---|---|---|
| Positive DN | 소교란 (scale=0.2) | GT class | 정확한 DOA + class 복원 |
| Negative DN | 대교란 (scale=0.8) | 무작위 셔플 | confidence=0 억제 |

- 프레임마다 **독립적인** 클래스 셔플 (`noise.argsort(dim=-1)`)
- `S_dn = n_dn_groups × 2 × n_slots` 개 DN 쿼리
- self-attention mask로 matching ↔ DN 쿼리 상호 차단

### 8.3 IterativeRefinementDecoder

DINO의 박스 delta 정밀화를 DOA 단위벡터 정밀화로 대응:

```
for each layer i in [0..3]:
    x = DecoderLayer(x, memory=7_tokens)   → [B*T, S_total, d]
    저장: layer_preds[i]

    if i < last_layer:
        # DOA 피드백 (matching 슬롯만)
        doa_raw  = doa_refine_head[i](x[:, :n_slots, :])  → [B*T, S, 3]
        doa_unit = normalize(doa_raw)                      → 단위벡터 투영
        pos_emb  = doa_pos_enc(doa_unit)                   → [B*T, S, d]
        x[:, :n_slots, :] += pos_emb                       → 다음 레이어 피드백
```

레이어별 DOA 위치 피드백이 실제 기하학적 정보를 전달하여 점진적 정밀화를 가능하게 한다.

### 8.4 DecoderLayer

```
x [B*T, S, d]      ← 쿼리
mem [B*T, 7, d]    ← 7 multi-scale 토큰

Self-attention   (S×S, + attn_mask for DN isolation) → post-norm
Cross-attention  (Q=x, KV=mem, 7개 토큰)              → post-norm
FFN (GELU)                                             → post-norm
```

---

## 9. Detection Heads

`sled/model/heads.py`

입력 `[B, T, S, d]` → 4개 출력:

| 헤드 | 구조 | 출력 | 의미 |
|---|---|---|---|
| `class_head` | Linear(d→n_classes) | `[B,T,S,209]` | 음원 클래스 로짓 (empty 없음) |
| `doa_head` | Linear(d→d→3) + L2 정규화 | `[B,T,S,3]` | DOA 단위벡터 |
| `loud_head` | Linear(d→1) | `[B,T,S]` | 음압 (dB) |
| `conf_head` | Linear(d→1) | `[B,T,S]` | Presence 로짓 |

> DINO와 달리 empty 클래스를 분류 헤드에서 분리 → 클래스 불균형 없음

---

## 10. Loss 함수

`sled/train.py`

### 10.1 Hungarian Matching

B×T 프레임 전체의 cost를 GPU에서 일괄 계산 후 CPU 1회 전송:

```
cost[i,j] = cls_cost[i,j] + 0.5 × doa_cost[i,j]

cls_cost = −softmax(logits_i)[cls_j]         ∈ [−1, 0]
doa_cost = 1 − cos_sim(pred_doa_i, gt_doa_j) ∈ [0,  2]
0.5 스케일로 범위 균형 조정
```

### 10.2 개별 Loss

#### Focal Loss (분류)
```
focal = α(1−p_t)^γ · BCE(logits, one_hot(target))
α=0.25, γ=2.0
```

#### DOA Loss
```
doa_loss = 1 − cos_sim(pred_doa, gt_doa)   ∈ [0, 2]
```

#### Presence Loss
```
conf_target = 1.0 (매칭된 슬롯) | 0.0 (비매칭)
presence_loss = BCE(conf_logit, conf_target)
```

### 10.3 전체 Loss 조합

```
total_loss = Σ_i  w_i × layer_loss(matching_preds_i, gt)
           + Σ_i  w_i × 0.5 × dn_loss(dn_preds_i, dn_targets)

layer weights = [0.2, 0.4, 0.6, 1.0]   (깊은 레이어 가중치 높음)

dn_loss = pos_loss + neg_conf_loss
  pos_loss     : positive DN → focal + doa + presence
  neg_conf_loss: negative DN → BCE(conf, 0)  (억제 학습)
```

---

## 11. 학습

`sled/train.py`

### 11.1 기본 설정

| 항목 | 값 |
|---|---|
| 옵티마이저 | AdamW (lr=1e-4, weight_decay=1e-4) |
| LR 스케줄 | CosineAnnealingLR (T_max=epochs, eta_min=1e-6) |
| 그래디언트 클리핑 | max_norm=1.0 |
| 배치 크기 | 8 |
| window_frames | 256 (5.12초) |
| n_classes | 209 |
| min_loudness_db | −60.0 dBFS |

### 11.2 학습 명령어

```bash
# 단일 데이터셋
python -m sled.train \
    --dataset-root    ./data \
    --sofa-path       ./hrtf/p0001.sofa \
    --epochs          200 \
    --batch-size      64 \
    --lr              3e-4 \
    --d-model         256 \
    --n-classes       209 \
    --window-frames   48 \
    --log-dir         ./runs \
    --checkpoint-dir  ./checkpoints

# 재개
python -m sled.train --resume checkpoints/sled_best.pt [옵션들]
```

**Linear Scaling Rule**: 배치를 K배 늘리면 lr도 K배.

### 11.2b 혼합 데이터셋 학습

두 데이터셋을 비율 조절하여 동시에 학습.

#### 주요 인수

| 인수 | 기본값 | 의미 |
|---|---|---|
| `--dataset-root2` | None | 2번 데이터셋 경로. 지정 시 혼합 활성화 |
| `--mix-ratio W1 W2` | 1.0 1.0 | 두 데이터셋 샘플링 비율 (합계≠1 허용, 자동 정규화) |
| `--val-mix-ratio VW1 VW2` | mix-ratio와 동일 | val 전용 비율. 생략 시 train 비율 그대로 |
| `--epoch-size N` | N1+N2 | train 에폭당 총 샘플 수 |
| `--val-size N` | N1_val+N2_val | val 에폭당 총 샘플 수 |

`--mix-ratio 0.7 0.3`의 의미: 매 배치에서 70%는 dataset1, 30%는 dataset2에서 샘플링.
각 데이터셋 내부에서는 모든 씬이 균등하게 뽑힘.

val은 `WeightedRandomSampler(replacement=False, generator=seed(0))`로 결정론적 평가.

#### 사용 예시

```bash
# 합성 7 : MRS balanced 3, val도 동일 비율
python -m sled.train \
    --dataset-root  ./data_custom_hrtf \
    --dataset-root2 ./data_mrs_balanced \
    --mix-ratio 0.7 0.3 \
    --n-classes 209 \
    --epochs 200

# train 7:3, val은 MRS만으로 평가
python -m sled.train \
    --dataset-root  ./data_custom_hrtf \
    --dataset-root2 ./data_mrs_balanced \
    --mix-ratio 0.7 0.3 \
    --val-mix-ratio 0 1 \
    --n-classes 209

# 에폭당 샘플 수 고정
python -m sled.train \
    --dataset-root  ./data_custom_hrtf \
    --dataset-root2 ./data_mrs_balanced \
    --mix-ratio 0.7 0.3 \
    --epoch-size 10000 \
    --val-size   1000
```

### 11.3 DN 커리큘럼

| 에폭 | n_dn_groups | 효과 |
|---|---|---|
| 1~30 | 5 | 많은 DN 쿼리로 GT 복원 집중 학습 |
| 31~ | 3 | 일반 탐지 학습 비중 증가 |

### 11.4 체크포인트

```
checkpoints/sled_epoch_XXXX.pt   10 에폭마다 저장
checkpoints/sled_best.pt         val_loss 최소 시 갱신
```

저장 내용: `{epoch, model, optimizer, scheduler, val_loss, use_hrtf_corr, use_ild, use_ipd}`

### 11.5 TensorBoard

```bash
tensorboard --logdir ./runs
```

| 태그 | 내용 |
|---|---|
| `Loss/train_step` | 배치별 train loss |
| `Loss/epoch` | 에폭별 train / val loss |
| `LR` | Learning rate |

---

## 12. 추론 및 시각화

`sled/visualize.py`

### 12.1 슬라이딩 윈도우 추론

```
오디오 전체 → window_frames × 960 단위 분할
→ 각 윈도우 모델 통과 (마지막 decoder layer 출력 사용)
→ 결과 concatenate → 전체 시퀀스
```

### 12.2 배치 시각화 스크립트

```bash
# scenes 0~9 생성 (기본)
bash make_viz.sh

# scenes 0~16 생성
bash make_viz.sh 0 16

# 씬 5 하나만
bash make_viz.sh 5 5
```

### 12.3 MP4 레이아웃 (1680 × 700 px)

```bash
# 합성 데이터셋 씬
python -m sled.visualize \
    --audio       data/audio/test/scene_011000.wav \
    --ckpt        checkpoints/sled_best.pt \
    --gt-json     data/audio/test/scene_011000.json \
    --class-map   data/meta/class_map.json \
    --output      output.mp4

# MRS mix 데이터셋 씬 (--class-map 생략 가능, 자동 탐색)
python -m sled.visualize \
    --audio       data_mrs_mix/audio/test/scene_006750.wav \
    --ckpt        checkpoints_mrs_mix_ft/sled_best.pt \
    --gt-json     data_mrs_mix/audio/test/scene_006750.json \
    --output      viz_mrs.mp4
```

**3-패널 구성**:

| 패널 | 내용 |
|---|---|
| **좌 — Polar (Top-Down)** | 극좌표 하향 투영. 반지름 = cos(el). 방위각 분포 파악. ●=GT, ★=예측 |
| **중 — Az-El Scatter** | 방위각(X) × 고도각(Y) 2D 산점도. 위/아래 구분 가능. el=0° 점선 표시 |
| **우 — Info Panel** | 슬롯별 class·az·el·confidence 수치 텍스트 |
| **하 — Waveform** | 4초 파형 스트립, 현재 시각 빨간선 |

> **개선**: 이전 버전 polar 단독 표시에서 az-el 2D 패널 추가.
> 고도각이 `cos(el)`로 매핑되면 +45°와 −45°가 동일 반지름에 표시되어 위/아래 구별 불가능했다. 중앙 패널에서 고도각 부호를 명확히 확인할 수 있다.

**GT 타이밍 정확도**: GT 프레임 겹침 판정이 `compute_dense_annotations`와 동일한 sample-based 로직 사용 (`start_sample < frame_end AND end_sample > frame_start`). 이전 time-based 체크에서 발생하던 0~20ms 타이밍 불일치 해소.

### 12.4 visualize.py 주요 수정 이력

| 수정 | 내용 |
|---|---|
| **n_classes 자동 감지** | `--n-classes` 미지정 시 checkpoint의 `heads.class_head.weight.shape[0]`에서 자동 추출. MRS finetune 체크포인트 등 다양한 클래스 수 지원 |
| **class_map 자동 탐색** | `--class-map` 미지정 시 `data/meta/class_map.json`, `data_test/meta/class_map.json` 순서로 자동 탐색 |
| **GT 없을 때 안전 처리** | `--gt-json` 미지정 시 legend에서 "Ground truth" 항목 자동 제거 |
| **파형 끊김 해소** | 매 프레임 `ax.cla()` 호출 대신 이전 artist(`fill_between`, `axvline`)를 `.remove()`로 개별 제거 → 축 재설정 없이 부드러운 업데이트 |
| **MRS-mix GT 지원** | JSON의 `azimuth=null`인 경우(`data_mrs_mix` 포맷) npy 어노테이션 파일에서 프레임별 DOA를 자동 로드. `load_gt_per_frame()` 내부에서 `annotations/<split>/` 경로 자동 추론 |

---

## 13. Ablation Study

입력 채널 각각의 기여도를 독립적으로 측정하기 위한 ablation 프레임워크.

### 13.1 제거 가능한 채널

| 플래그 | 제거 채널 | 파라미터 변화 |
|---|---|---|
| `--no-ild` | ILD (ch2) | −576 (stem3 weight 1채널) |
| `--no-ipd` | sin/cos IPD (ch3·4) | −1,152 (stem3 weight 2채널) |
| `--no-hrtf-corr` | HRTF 히트맵 | −590,848 (HRTFProjection 전체) |

채널 제거 시 `in_channels`가 줄어들며 해당 conv weight도 제거 → dead weight 없음.

### 13.2 학습 명령어

```bash
# 기본 (전체 채널)
python -m sled.train --checkpoint-dir checkpoints/full ...

# 각 채널 제거 ablation
python -m sled.train --no-ild         --checkpoint-dir checkpoints/no_ild   ...
python -m sled.train --no-ipd         --checkpoint-dir checkpoints/no_ipd   ...
python -m sled.train --no-hrtf-corr   --checkpoint-dir checkpoints/no_hrtf  ...

# 중복 제거 (L/R mel만 남는 베이스라인)
python -m sled.train --no-ild --no-ipd --no-hrtf-corr \
                     --checkpoint-dir checkpoints/no_binaural ...
```

### 13.3 평가 스크립트

`sled/eval.py`: 체크포인트 → test split에서 4가지 지표 계산

| 지표 | 의미 |
|---|---|
| `test_loss` | Hungarian matching loss (학습 loss와 동일) |
| `det_f1` | 프레임 단위 음원 탐지 F1 (P·R 포함) |
| `cls_acc` | Hungarian 매칭된 (예측, GT) 쌍의 클래스 정확도 |
| `doa_mae` | 매칭된 쌍의 평균 각도 오차 (도) |
| `ms/frame` | 모델 순전파 시간 (ms/annotation frame) |

추론 시간: CUDA 환경에서 `torch.cuda.Event`로 GPU 정확도 측정. Warmup 1회로 JIT 오버헤드 제거.

### 13.4 Ablation 자동화 스크립트

```bash
# ablation.sh 상단에서 각 config의 체크포인트 경로 설정
CKPT_full="/path/to/full/sled_best.pt"
CKPT_no_ild="/path/to/no_ild/sled_best.pt"
CKPT_no_ipd="/path/to/no_ipd/sled_best.pt"
CKPT_no_hrtf="/path/to/no_hrtf/sled_best.pt"
CKPT_no_binaural="/path/to/no_binaural/sled_best.pt"

# eval-only 실행
bash ablation.sh

# 특정 config만
bash ablation.sh --configs "full no_hrtf"

# train + eval (처음부터 학습)
bash ablation.sh --train --epochs 200
```

결과는 `ablation_results.jsonl`에 누적 저장되고, 마지막에 비교 테이블 출력:

```
  Config         ILD IPD HRTF     Loss    ΔLoss  Det-F1  Cls-Acc  DOA-MAE  ms/frame
  ─────────────────────────────────────────────────────────────────────────────────
  full            Y   Y   Y    x.xxxx       —   x.xxxx   x.xxxx   xx.xx°   x.xxxms
  no_ild          Y   Y   N    x.xxxx  +x.xxxx  x.xxxx   x.xxxx   xx.xx°   x.xxxms
  no_ipd          Y   N   Y    x.xxxx  +x.xxxx  x.xxxx   x.xxxx   xx.xx°   x.xxxms
  no_hrtf         N   Y   Y    x.xxxx  +x.xxxx  x.xxxx   x.xxxx   xx.xx°   x.xxxms
  no_binaural     N   N   N    x.xxxx  +x.xxxx  x.xxxx   x.xxxx   xx.xx°   x.xxxms
```

### 13.5 MRS 체크포인트 비교 스크립트

`compare_mrs_ckpts.sh`: 두 체크포인트를 `data_mrs_mix` test 50개 씬으로 비교.

```bash
# 스크립트 상단에서 체크포인트 경로 설정 후 실행
bash compare_mrs_ckpts.sh
```

결과(`compare_mrs_results.jsonl`)에 두 체크포인트의 지표와 차이(Δ, ↑/↓)가 표시:

```
  Checkpoint              Loss     Det-F1   Cls-Acc   DOA-MAE   avg/win   p99/win       RTF
  checkpoints_mrs_mix_ft  x.xxxx   x.xxxx   x.xxxx   xx.xx°    x.xxms   xx.xxms   x.xxxxx
  checkpoints_mrs_mix2    x.xxxx   ...      (Δ 표시)
```

---

## 14. 파일 구조

```
crossCorr/
├── hrtf/p0001.sofa              HRTF (SOFA 형식)
├── soud_effects/                음원 클립 (FSD50K 심볼릭 링크)
├── data/                        합성 데이터셋 (build_dataset.py 출력)
│   ├── meta/{class_map.json, split.json}
│   ├── audio/{train,val,test}/
│   └── annotations/{train,val,test}/
├── data_mrs_mix/                MRS binaural 혼합 데이터셋 (build_mrs_mix_dataset.py 출력)
│   ├── meta/{class_map.json, split.json}
│   ├── audio/{train,val,test}/   scene_NNNNNN.wav + .json
│   └── annotations/{train,val,test}/  scene_NNNNNN_{cls,doa,loud,mask}.npy
├── data_mrs_balanced/           azimuth 균등 MRS 데이터셋 (build_mrs_balanced_dataset.py 출력)
│   └── (data_mrs_mix와 동일 구조)
├── MRSAudio/MRSLife/MRSSound/   원본 MRS binaural 녹음 (sound001~sound208)
├── checkpoints*/                학습 체크포인트
├── runs*/                       TensorBoard 로그
├── make_viz.sh                  배치 시각화 생성 스크립트
├── ablation.sh                  채널 ablation 자동화 스크립트
├── ablation_results.jsonl       ablation 결과 누적 파일
├── build_mrs_mix_dataset.py     MRS binaural → 다중음원 씬 합성기 (DOA 편향 있음)
├── build_mrs_balanced_dataset.py MRS → azimuth 균등 다중음원 씬 합성기
├── compare_mrs_ckpts.sh         MRS 체크포인트 2개 비교 스크립트
└── sled/
    ├── model/
    │   ├── preprocessor.py      AudioPreprocessor (use_ild, use_ipd, use_hrtf_corr)
    │   ├── encoder.py           SLEDEncoder, HRTFProjection (조건부), CausalBiFPN
    │   ├── decoder.py           CrossAttentionQuerySelector, ContrastiveDeNoising,
    │   │                        IterativeRefinementDecoder
    │   ├── heads.py             DetectionHeads
    │   └── sled.py              SLEDv3 (use_ild, use_ipd, use_hrtf_corr 플래그)
    ├── dataset/
    │   ├── synthesizer.py       씬 합성기 (합성 데이터셋용)
    │   ├── build_dataset.py     데이터셋 빌더 (멀티프로세싱)
    │   ├── torch_dataset.py     SLEDDataset + build_dataloader + build_mixed_dataloader
    │   └── mrs_dataset.py       MRS 단일음원 데이터셋 (train_mrs.py용)
    ├── train.py                 학습 스크립트 (단일/혼합 데이터, --mix-ratio, --val-mix-ratio 등)
    ├── train_mrs.py             MRS 단일음원 전용 학습 스크립트
    ├── eval.py                  평가 스크립트 (loss/F1/acc/DOA/time)
    ├── stream_bench.py          스트리밍 레이턴시 + 성능 벤치마크 (B=1)
    └── visualize.py             MP4 시각화 (1680px, polar + az-el 패널)
```

---

## 15. 주요 하이퍼파라미터

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `d_model` | 256 | 트랜스포머 feature 차원 |
| `n_slots` | 3 | 프레임당 최대 동시 음원 수 |
| `n_classes` | 209 | 음원 클래스 수 (empty 제외) |
| `n_decoder_layers` | 4 | IterativeRefinement 레이어 수 |
| `n_conformer_layers` | 4 | CausalConformer 블록 수 |
| `n_bifpn` | 2 | CausalBiFPN 반복 수 |
| `n_dn_groups` | 3~5 | DN 그룹 수 (커리큘럼) |
| `window_frames` | 48 | 학습 윈도우 길이 (×20ms = 5.12s) |
| `hop_length` | 960 | STFT hop = 어노테이션 프레임 단위 |
| `min_loudness_db` | −60.0 | 비활성 처리 음량 임계값 (dBFS) |
| `use_ild` | True | ILD 채널 사용 여부 (ablation) |
| `use_ipd` | True | IPD 채널 사용 여부 (ablation) |
| `use_hrtf_corr` | True | HRTF 히트맵 사용 여부 (ablation) |
