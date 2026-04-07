# SLED Dataset — Binaural Synthetic Dataset

> 생성 기준: 2026-04-06 (v2: 연속 위치 샘플링 + SRIR)
> 학습 대상: SLED (Sound Localization & Event Detection) 모델
> 입력 형식: Stereo binaural 44.1→48kHz, 2채널 WAV
> 목표 규모: Train 10,000 + Val 1,000 + Test 500 scenes × 45s = **144시간**

---

## 1. 디렉토리 구조

```
dataset/
├── audio/
│   ├── train/   scene_000000.wav … scene_009999.wav   (stereo 16-bit 48 kHz)
│   ├── val/     scene_010000.wav … scene_010999.wav
│   └── test/    scene_011000.wav … scene_011499.wav
│
├── annotations/              (JSON — 디버깅/시각화용)
│   ├── train/   scene_000000.json …
│   ├── val/
│   └── test/
│
├── annotations_dense/        (numpy binary — 학습용 고속 I/O)
│   ├── train/
│   │   ├── scene_000000_cls.npy    [T, 5]     int16   class_id (-1=empty)
│   │   ├── scene_000000_doa.npy    [T, 5, 3]  float16 unit vector (x,y,z)
│   │   ├── scene_000000_loud.npy   [T, 5]     float16 dBFS per source
│   │   └── scene_000000_mask.npy   [T, 5]     bool    slot active
│   ├── val/
│   └── test/
│
├── meta/
│   ├── class_map.json        class_id(0-199) → label, + 300="__empty__"
│   ├── hrtf_registry.json    HRTF subject list with n_positions, sample_rate
│   ├── split.json            train/val/test scene_id lists
│   └── progress_{split}.txt  합성 진행 체크포인트 (자동 생성)
│
├── sources/                  (원본 데이터 — 합성에 사용, 변경 없음)
│   ├── HRTF/                 SONICOM HRTF (140 subjects, .sofa)
│   ├── SRIR/
│   │   ├── TAU-SRIR_DB/      9 rooms × conditions (.mat HDF5)
│   │   └── TAU-SNoise_DB/    Room ambience FOA recordings
│   └── sound_effects/        FSD50K (200 classes, ~51K clips)
│
├── synthesizer/              합성 코드 (아래 §3 참조)
└── torch_dataset.py          PyTorch DataLoader
```

---

## 2. Scene 구성

### 2.1 오디오

| 항목 | 값 |
|---|---|
| Sample rate | 48,000 Hz |
| Channels | 2 (binaural L/R) |
| Bit depth | 16-bit PCM WAV |
| Scene 길이 | 45 s (2,160,000 samples) |
| Frame hop | 20 ms (960 samples) → T = 2,250 frames |

### 2.2 음원 수 분포

| 음원 수 | 비율 |
|---|---|
| 0 (ambient only) | 5% |
| 1 | 25% |
| 2 | 30% |
| 3 | 25% |
| 4 | 10% |
| 5 | 5% |

1~3개가 80%로 가장 많음 (Curriculum 학습에 유리).

### 2.3 음원 위치

- **Azimuth**: 연속 균등 랜덤 [−180°, +180°] (SLED CW: 0=front, +=right)
- **Elevation**: 연속 균등 랜덤 [−45°, +45°]
- 위치는 SOFA 측정 그리드에 구속되지 않음 — HRTFInterpolator(IDW)로 임의 방향 보간
- 이동 음원: 현재 구현은 정적 위치 (moving source 지원 예정)

### 2.4 Onset / Offset

- 약 40% 확률로 씬 내 일부 구간만 활성화
- 최소 2초 / 최대 30초

---

## 3. 합성 파이프라인

### 3.1 흐름도 (v2: 연속 위치 + SRIR)

```
SFX mono clip
  │   (resample to 48 kHz, loop/crop)
  │
  ├─── 1. 위치 배정: (az, el) 연속 균등 랜덤 샘플링
  │         az ∈ [−180°, +180°]  el ∈ [−45°, +45°]
  │
  ├─── 2. Loudness GT 계산 ← dry mono per-frame RMS → dBFS
  │
  ├─── 3. HRTF 보간 (binaural_engine.HRTFInterpolator, IDW, K=3)
  │         → 정확한 연속 위치에서의 HRIR (hrir_l, hrir_r)
  │
  ├─── 4. SRIR 룩업 (TAU-SRIR_DB, W-channel)
  │         → 씬당 랜덤 room + RT60 + dist 조건 선택
  │         → 원형 room: 가장 가까운 1° 단위 az의 W-channel
  │            비원형 room: 랜덤 trajectory position의 W-channel
  │         → srir_w (14,400 samples @ 48 kHz)
  │
  ├─── 5. BRIR 합성
  │         BRIR_L = conv(srir_w, hrir_l)   ← 방향: HRTF / 잔향: SRIR
  │         BRIR_R = conv(srir_w, hrir_r)
  │
  └─── 6. Overlap-Add 합성 (oaconvolve)
          → binaural spatialized source

All sources → Mix (sum)
  │
  └─── 7. Peak normalize → stereo WAV + JSON + dense .npy 저장
          (별도 노이즈 추가 없음 — SRIR 잔향 꼬리에 room noise 포함)
```

### 3.2 연속 위치 렌더링 원리

**방향 (HRTF)**:  
`binaural_engine.HRTFInterpolator`가 임의의 (az, el)에서 주파수 영역 IDW 보간으로 HRIR을 계산한다.
- K=3 최근접 측정 방향 탐색 (geodesic 거리)
- log-magnitude 가중 평균 + nearest-phase → 위상 smearing 없음
- SOFA 그리드에 구속되지 않아 연속적 방향 표현 가능

**공간 음향 (SRIR)**:  
TAU-SRIR_DB의 FOA W-channel (무지향성)을 room acoustics 필터로 사용한다.
- W-channel = 0차 ambisonics (omnidirectional) → 방향 편향 없이 잔향만 제공
- 씬당 1개 room+조건 공유 → 물리적 일관성
- SRIR 측정 자체에 room ambient noise 포함 → 별도 노이즈 불필요

**BRIR = conv(SRIR_W, HRIR)**:  
- HRTF가 정확한 연속 방향 단서 (ITD, ILD, spectral notch) 제공
- SRIR이 실제 room의 잔향·초기 반사 제공
- BRIR 길이 ≈ 14,400 + 256 − 1 = 14,655 samples

### 3.3 Loudness Ground Truth

```
per_source_loudness[frame] = 20 × log10( RMS( mono_dry[frame×hop : (frame+1)×hop] ) )
```

- **mix된 binaural이 아닌** 각 음원의 dry mono 기준
- 씬 합성 전에 계산하므로 공간화 효과에 무관
- Inactive frame → NaN (dense .npy에서 mask==False인 slot)

---

## 4. 데이터 소스

### 4.1 SONICOM HRTF Dataset

| 항목 | 값 |
|---|---|
| Subjects | 140개 (p0001.sofa ~ p0205.sofa, hrtf/ 디렉터리) |
| 씬당 선택 방식 | **씬마다 140개 중 랜덤 1명** 선택 → HRTF 다양성 확보 |
| Positions per subject | 828 |
| HRIR length | 256 samples (at 48 kHz = 5.3 ms) |
| Sample rate | 48,000 Hz |
| Format | SOFA (HDF5) |
| 보간 방식 | HRTFInterpolator (IDW, K=3) → 임의 연속 위치에서 HRIR 합성 |
| 좌표 | azimuth 0–360° CCW (SOFA) → SLED CW 변환: az_sofa = -az_sled |

### 4.2 TAU-SRIR Database

| 항목 | 값 |
|---|---|
| Rooms | 9개 (bomb_shelter, gym, pb132, pc226, sa203, sc203, se203, tb103, tc352) |
| 원본 SR | 24,000 Hz → 48,000 Hz 리샘플 |
| FOA channels | 4 (ACN order: W, Y, Z, X) |
| RIR length | 7,200 samples @ 24 kHz → 14,400 @ 48 kHz (300 ms) |
| Conditions | 156개 (room × RT60 × source_distance) |
| 원형 rooms (360 az) | bomb_shelter, gym, pb132, pc226, tc352 |
| 비원형 rooms | sa203 (72), sc203 (61), se203 (93), tb103 (84) |

**TAU-SNoise_DB**: 각 room에 대응하는 FOA ambient noise (~28분 길이, 24 kHz)
합성 시 partial read로 필요한 구간만 로드해 I/O 최소화.

### 4.3 FSD50K

| 항목 | 값 |
|---|---|
| 클래스 수 | 200 (vocabulary.csv, index 0–199) |
| 총 클립 수 | ~51,000 (dev 40,966 + eval 10,231) |
| 활성 클래스 | 199개 (모든 파일 존재 확인 후) |
| 오디오 형식 | 다양한 SR → 48 kHz 리샘플, 모노 다운믹스 |

---

## 5. Annotation 스키마

### 5.1 JSON (annotations/{split}/scene_{id}.json)

```json
{
  "scene_id": "000000",
  "audio_file": "audio/train/scene_000000.wav",
  "sample_rate": 48000,
  "duration_sec": 45.0,
  "synthesis_meta": {
    "room": "bomb_shelter",
    "srir_rt60_idx": 3,
    "srir_dist_idx": 0,
    "hrtf_subject": "p0042",
    "snr_db": 14.7
  },
  "frame_config": {
    "hop_sec": 0.02,
    "total_frames": 2250
  },
  "sources": [
    {
      "source_id": 0,
      "class_id": 42,
      "class_name": "Dog",
      "onset_frame": 100,
      "offset_frame": 1800,
      "trajectory": [
        {"frame": 100, "azimuth_deg": 45.0, "elevation_deg": 10.0},
        {"frame": 950, "azimuth_deg": -30.0, "elevation_deg": 5.0},
        {"frame": 1800, "azimuth_deg": -30.0, "elevation_deg": 5.0}
      ]
    }
  ]
}
```

### 5.2 Dense Binary (annotations_dense/{split}/scene_{id}_*.npy)

| 파일 | Shape | Dtype | 설명 |
|---|---|---|---|
| `_cls.npy` | [T, 5] | int16 | class_id, -1=inactive |
| `_doa.npy` | [T, 5, 3] | float16 | (x, y, z) unit vector |
| `_loud.npy` | [T, 5] | float16 | dBFS of dry mono per frame |
| `_mask.npy` | [T, 5] | bool | slot active flag |

**Slot 배정**: onset_frame 기준 오름차순 정렬 (Hungarian matching은 학습 시 수행).

**DOA unit vector 변환**:
```
x = cos(el) × cos(az)    (forward)
y = cos(el) × sin(az)    (right, az+=right)
z = sin(el)              (up)
```
학습 시 역변환: `azimuth = atan2(y, x)`, `elevation = asin(z)`

---

## 6. 합성 코드 구조

```
dataset/synthesizer/
├── config.py              SynthConfig dataclass (paths, params)
├── hrtf_loader.py         HRTFLibrary: SONICOM SOFA 로더 + KD-tree 위치검색
├── srir_loader.py         SRIRLibrary: TAU-SRIR_DB 로더 + azimuth 매핑
├── fsd50k_loader.py       FSD50KCatalog: 클래스별 샘플링, 리샘플, 정규화
├── binaural_render.py     BinauralRenderer: BF filter 계산, BRIR 합성
├── scene_synth.py         synthesize_scene(): 씬 합성 오케스트레이터
├── annotation_writer.py   write_annotations(): JSON + .npy 저장
├── build_meta.py          class_map.json, hrtf_registry.json 생성
├── run_synthesis.py       multiprocessing 메인 (spawn Pool)
├── verify_dataset.py      단일 씬 합성 + 검증 스크립트
└── requirements.txt       numpy, scipy, soundfile, h5py, tqdm, torch
```

### 핵심 모듈 역할

**`binaural_render.py`** — 물리 기반 binaural 렌더링
- `compute_binaural_foa_filters(hrtf_subject, n_vls=50)` → BF `(4, 2, 256)`
  - HRTF subject당 1회만 계산, scene 합성 중 재사용
- `build_brir(srir_foa, bf)` → BRIR `(2, 14655)`: 8번의 FFT 합성곱
- `spatialize(mono, brir)` → binaural: **`oaconvolve`** 사용 (긴 신호 × 짧은 필터에 최적)

**`synthesizer.py`** — 씬 합성 (v2)
- `synthesize_scene()`: SOFA HRTF + TAU-SRIR + SFX를 조합해 1개 씬 생성
- 연속 (az, el) 샘플링 → HRTFInterpolator로 보간 → 오차 없는 DOA annotation
- 씬당 1개 room+조건 공유, 소스별 독립적 연속 방향
- SRIR W-channel을 room acoustics 필터로 사용 (노이즈 별도 추가 없음)

**`run_synthesis.py`** — 병렬 합성
- `mp.Pool(spawn)`: 각 워커가 독립적인 HRTF/SRIR/FSD50K 인스턴스 소유
- worker init에서 HRTF 전체 pre-load (~450 MB/worker)
- 씬ID ↔ seed 결정적 매핑 → 완전 재현 가능, 부분 재생성 가능
- `progress_{split}.txt` 체크포인트 → `--resume`으로 중단 후 재개

---

## 7. 학습용 PyTorch DataLoader

```python
# dataset/torch_dataset.py
from dataset.torch_dataset import build_dataloader

train_loader = build_dataloader(
    dataset_root="/path/to/dataset",
    split="train",
    batch_size=32,
    window_frames=256,      # 256 × 20ms = 5.12 s 랜덤 윈도우
    augment_scs=True,       # Stereo Channel Swap (L↔R + azimuth 부호 반전)
    num_workers=16,
    prefetch_factor=4,
    pin_memory=True,
)
```

**반환 item:**

| Key | Shape | Dtype | 설명 |
|---|---|---|---|
| `audio` | [2, N] | float32 | stereo waveform (5.12 s) |
| `cls` | [T, 5] | int64 | class ids |
| `doa` | [T, 5, 3] | float32 | DOA unit vectors |
| `loud` | [T, 5] | float32 | dBFS |
| `mask` | [T, 5] | bool | active slot mask |
| `scene_id` | str | — | 디버깅용 |

`audio` → 모델 preprocessor에서 5채널 변환:
```python
# SLED 모델 입력: [B, 5, 64, T]
# [L-mel, R-mel, cos(IPD), sin(IPD), ILD]
```

**Data Augmentation (DataLoader 내):**

| 기법 | 위치 | 적용률 |
|---|---|---|
| **SCS** (Stereo Channel Swap) | `torch_dataset.py` | 50% (train) |
| **SpecAugment** (time/freq mask) | 모델 preprocessor 이후 | 학습 중 적용 권장 |
| **Mixup** | Trainer 레벨 | 선택 사항 |

---

## 8. 합성 실행

```bash
cd /home/rllab/Desktop/crossCorr

# ── 전체 데이터셋 합성 (10k train + 1k val + 500 test, 16 workers) ──────────
python -m sled.dataset.build_dataset \
    --output-dir ./data \
    --num-train 10000 --num-val 1000 --num-test 500 \
    --workers 16

# ── 빠른 smoke test (씬 10개, 1 worker) ──────────────────────────────────────
python -m sled.dataset.build_dataset \
    --output-dir ./data_test \
    --num-train 10 --num-val 2 --num-test 2 \
    --workers 1

# ── 중단 후 재개 ──────────────────────────────────────────────────────────────
python -m sled.dataset.build_dataset \
    --output-dir ./data \
    --workers 16 --resume

# ── 커스텀 경로 지정 ──────────────────────────────────────────────────────────
python -m sled.dataset.build_dataset \
    --output-dir  ./data \
    --hrtf-dir    ./hrtf \
    --sfx-dir     ./soud_effects \
    --srir-dir    ./sources/TAU_SRIR/TAU-SRIR_DB \
    --workers 16
```

**성능 (30초 씬 기준 추정):**

| 항목 | 값 |
|---|---|
| 30초 씬 합성 시간 (single core) | ~3–8 sec (SOFA 로드 포함) |
| SOFA 로드 overhead (씬당) | ~0.3 sec (HRTFInterpolator 생성) |
| 16 워커 처리량 | ~2–5 scenes/sec |
| 전체 11,500 씬 예상 시간 | ~30–90분 |

---

## 9. 데이터 규모

| 구성 | 수량 | 오디오 용량 |
|---|---|---|
| Train | 10,000 scenes × 45 s | ~86 GB |
| Val | 1,000 scenes | ~8.6 GB |
| Test | 500 scenes | ~4.3 GB |
| Dense annotations | 11,500 scenes | ~1.4 GB |
| JSON annotations | 11,500 scenes | ~57 MB |
| **총계** | **144시간** | **~100 GB** |

---

## 10. 주의사항 및 설계 결정

### Loudness GT
합성 전 **dry mono 기준** RMS를 측정. 공간화(잔향, HRTF 레벨 변화)에 의존하지 않아야 SLED 모델이 intrinsic loudness를 학습할 수 있음.

### SRIR 좌표 규약
- TAU-SRIR DB: 원형 room의 360개 position = azimuth 0°~359° (1° 간격, CCW)
- 소스 az (SLED CW) → SRIR index 변환: `az_idx = round((-az_sled) % 360) % 360`
- 비원형 room (sa203 등): trajectory position, 좌표 불명 → 랜덤 선택
- SRIR 잔향이 방향 정보를 포함하지만, W-channel 사용으로 directional bias 최소화

### HRTF 좌표 규약
- SOFA azimuth: 0=front, 90=left (CCW) ↔ SLED CW: `az_sofa = -az_sled`
- HRTFInterpolator: K=3 IDW 보간 → 임의 연속 위치에서 HRIR 합성
- annotation DOA = HRTF에 사용된 정확한 (az, el) 기준 → 오차 없음

### on-the-fly vs pre-synthesized
현재 구현은 **pre-synthesized** (디스크에 미리 저장).
학습 중 on-the-fly 합성은 CPU 병목(특히 HRTF pre-load ~450 MB/worker)을 야기하므로 권장하지 않음.
데이터 다양성 증대가 필요하면 pre-synthesized 데이터를 추가로 생성할 것.

### FSD50K 클래스 → SLED 클래스
FSD50K: 200 클래스 (ID 0–199)
SLED 모델 설계: 300 클래스 + 1 empty
→ 현재 데이터셋은 FSD50K 200 클래스 사용. 추가 데이터소스(AudioSet segments 등) 확보 시 301개 full class 활용 가능.

### 이동 음원 렌더링
현재 구현: 정적 위치 (씬 내 동일 az/el). 이동 음원 지원 예정.
연속 위치이므로 임의의 고밀도 waypoint도 HRTFInterpolator로 직접 보간 가능.

### SRIR + HRTF 결합 방식 (v2)
- **W-channel 전용**: FOA의 0차 채널 (omnidirectional) 만 사용
  - 방향 편향 없이 room reverb character만 제공
  - 방향 단서는 100% HRTFInterpolator가 담당
- **오차 없는 DOA annotation**: 샘플된 (az, el)이 HRTF에 직접 사용됨
  (구 SOFA nearest-neighbor 방식의 discrete position 오차 제거)
