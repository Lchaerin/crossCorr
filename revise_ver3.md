# SLED v3 → v3.1 Revision Guide (Multi-Source DOA 성능 개선)

> **목적**: 단일 음원 대비 다중 음원 환경에서 DOA 추정 성능이 급격히 저하되는 문제 해결.
> 모델 크기 증가 허용. 아래 3가지 핵심 수정을 적용할 것.

---

## 수정 개요

| # | 수정 사항 | 파일 | 난이도 |
|---|----------|------|--------|
| 1 | Per-frame HRTF injection | `preprocessor.py`, `encoder.py`, `sled.py` | 중 |
| 2 | Query Selector에 slot spatial prior 추가 | `decoder.py` | 하 |
| 3 | Sub-band memory 세분화 (6→18 tokens) | `encoder.py`, `sled.py` | 중 |

---

## 수정 1: Per-frame HRTF Injection

### 문제
현재 HRTF cross-correlation heatmap은 윈도우 전체에서 8프레임을 샘플링해 평균한 뒤
단일 global embedding `[B, d_model]`로 만들어 encoder 출력에 broadcast 더한다.
다중 음원이 시간에 따라 이동하거나 on/off되면 이 평균이 spatial cue를 파괴한다.

### 변경 내용

#### `preprocessor.py` — `AudioPreprocessor.forward()`

현재 `ch5`는 `[B, 64, 32]` 고정 크기 (윈도우 평균). 이를 **per-frame**으로 변경한다.

**변경 전 (line ~225-250, HRTF cross-correlation heatmap 블록):**
```python
# Sample 8 evenly spaced frames across the window and average → [B, F]
T_stft    = csd_full.shape[-1]
idx8      = torch.linspace(0, T_stft - 1, 8).long()
csd_r_avg = csd_full.real[..., idx8].mean(dim=-1)
csd_i_avg = csd_full.imag[..., idx8].mean(dim=-1)
# ... (이하 correlation → scatter_add → ch5 [B, az_bins, el_bins])
```

**변경 후:**
```python
# Per-frame HRTF: compute correlation for EVERY STFT frame
# csd_full: [B, F, T_stft] complex
T_stft = csd_full.shape[-1]

# Per-direction correlation for all frames at once
# W_real: [N_DIR, F], csd_full.real: [B, F, T_stft]
# einsum → [B, N_DIR, T_stft]
corr_unnorm = (
    torch.einsum('df,bft->bdt', self.W_real, csd_full.real) -
    torch.einsum('df,bft->bdt', self.W_imag, csd_full.imag)
)
# Normalisation per frame
norm1_sq = torch.einsum('df,bft->bdt', self.norm_hr_sq, pow_L)  # [B, N_DIR, T_stft]
norm2_sq = torch.einsum('df,bft->bdt', self.norm_hl_sq, pow_R)
corr = corr_unnorm / (norm1_sq * norm2_sq + 1e-8).sqrt()        # [B, N_DIR, T_stft]

# Build 2D az × el grid per frame [B, T_stft, az_bins, el_bins]
az_bins = self.n_mels    # 64
el_bins = 32

el_bin_float = (self.elevations + 90.0) / 180.0 * el_bins
el_bin_idx   = el_bin_float.long().clamp(0, el_bins - 1)
flat_idx     = self.az_bin_idx * el_bins + el_bin_idx              # [N_DIR]
flat_idx_b   = flat_idx.view(1, -1, 1).expand(B, -1, T_stft)      # [B, N_DIR, T_stft]

# Transpose corr to [B, T_stft, N_DIR] for scatter
corr_t     = corr.permute(0, 2, 1).reshape(B * T_stft, -1)         # [B*T, N_DIR]
flat_idx_t = flat_idx.view(1, -1).expand(B * T_stft, -1)           # [B*T, N_DIR]

hrtf_flat  = corr_t.new_zeros(B * T_stft, az_bins * el_bins)
count_flat = corr_t.new_zeros(B * T_stft, az_bins * el_bins)
hrtf_flat.scatter_add_(1, flat_idx_t, corr_t)
count_flat.scatter_add_(1, flat_idx_t, torch.ones_like(corr_t))

ch5 = (hrtf_flat / (count_flat + 1e-8)).view(B, T_stft, az_bins, el_bins)
# ch5 shape: [B, T_stft, 64, 32]  (was [B, 64, 32])
```

**반환값 변경:** `return out, ch5` — ch5 shape이 `[B, T, 64, 32]`로 바뀜.

---

#### `encoder.py` — `HRTFProjection` 클래스

**변경 전:**
```python
class HRTFProjection(nn.Module):
    def __init__(self, az_bins=64, el_bins=32, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(az_bins * el_bins, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, ch5):
        return self.proj(ch5)   # [B, d_model]
```

**변경 후:**
```python
class HRTFProjection(nn.Module):
    """Maps per-frame HRTF heatmap [B, T, 64, 32] → [B, T, d_model].

    Uses a small 2D CNN to preserve spatial structure before projecting.
    """

    def __init__(self, az_bins: int = 64, el_bins: int = 32, d_model: int = 256):
        super().__init__()
        # 2D CNN: treats each frame's [64, 32] heatmap as a 1-channel image
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → [32, 16]
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # → [16, 8]
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                                 # → [1, 1]
        )
        self.proj = nn.Sequential(
            nn.Linear(128, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, ch5: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ch5 : [B, T, 64, 32]  per-frame HRTF heatmap

        Returns
        -------
        [B, T, d_model]
        """
        B, T, H, W = ch5.shape
        x = ch5.reshape(B * T, 1, H, W)        # [B*T, 1, 64, 32]
        x = self.cnn(x).squeeze(-1).squeeze(-1) # [B*T, 128]
        x = self.proj(x)                         # [B*T, d_model]
        return x.reshape(B, T, -1)               # [B, T, d_model]
```

---

#### `encoder.py` — `SLEDEncoder.forward()`

HRTF injection 부분 변경:

**변경 전 (line ~240-242):**
```python
if hrtf_ch is not None:
    hrtf_feat = self.hrtf_proj(hrtf_ch)   # [B, d]
    x = x + hrtf_feat.unsqueeze(1)          # [B, T, d]
```

**변경 후:**
```python
if hrtf_ch is not None:
    # hrtf_ch: [B, T_stft, 64, 32] — T_stft may differ from T after conv
    # Align temporal dimension to match encoder output T
    T_enc = x.shape[1]
    if hrtf_ch.shape[1] != T_enc:
        # Linear interpolate along time axis
        hrtf_ch = hrtf_ch.permute(0, 2, 3, 1)  # [B, 64, 32, T_stft]
        hrtf_ch = F.interpolate(
            hrtf_ch.reshape(B, 64 * 32, -1),    # [B, 2048, T_stft]
            size=T_enc, mode='nearest'
        ).reshape(B, 64, 32, T_enc).permute(0, 3, 1, 2)  # [B, T_enc, 64, 32]
    hrtf_feat = self.hrtf_proj(hrtf_ch)   # [B, T, d]
    x = x + hrtf_feat                      # [B, T, d]  (per-frame addition)
```

> **주의:** `SLEDEncoder.__init__`에서 `B`가 `forward` 스코프에 없으므로 `B = x.shape[0]`을 해당 블록 전에 추가할 것.

---

#### `sled.py` — `SLEDv3.forward()`

`hrtf_ch` shape 관련 주석만 업데이트하면 됨 (preprocessor → encoder로 그대로 전달).

```python
# 변경 전 주석:
# feat: [B, 5, 64, T]   hrtf_ch: [B, 64, 32]

# 변경 후 주석:
# feat: [B, 5, 64, T]   hrtf_ch: [B, T_stft, 64, 32]
```

---

## 수정 2: Query Selector에 Slot Spatial Prior 추가

### 문제
`CrossAttentionQuerySelector`의 `slot_queries`가 `torch.randn`으로만 초기화되어
모든 슬롯이 동일한 dominant source에 collapse하기 쉽다.

### 변경 내용

#### `decoder.py` — `CrossAttentionQuerySelector.__init__()`

**변경 전 (line 65-67):**
```python
self.slot_queries = nn.Parameter(
    torch.randn(n_slots, d_model) * 0.02
)
```

**변경 후:**
```python
# Spatial prior: each slot starts with a different angular bias
# For n_slots=3 → 0°, 120°, 240° initial bias
import math
slot_init = torch.randn(n_slots, d_model) * 0.02
# Encode angular prior into first 3 dimensions
for s in range(n_slots):
    angle = 2 * math.pi * s / n_slots
    slot_init[s, 0] += 0.5 * math.cos(angle)
    slot_init[s, 1] += 0.5 * math.sin(angle)
    slot_init[s, 2] += 0.0  # elevation neutral
self.slot_queries = nn.Parameter(slot_init)

# Slot-specific spatial embedding (learnable, added after cross-attn)
self.slot_spatial_embed = nn.Parameter(
    torch.randn(n_slots, d_model) * 0.02
)
```

#### `decoder.py` — `CrossAttentionQuerySelector.forward()`

**변경 전 (line 100-103):**
```python
attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
q = self.norm1(q + attn_out)
q = self.norm2(q + self.ffn(q))
return q
```

**변경 후:**
```python
attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
q = self.norm1(q + attn_out)

# Add slot-specific spatial embedding to encourage diversity
spatial_emb = self.slot_spatial_embed.unsqueeze(0).expand(B * T, -1, -1)
q = q + spatial_emb

q = self.norm2(q + self.ffn(q))
return q
```

---

## 수정 3: Sub-band Memory 세분화 (6→18 tokens)

### 문제
현재 P3/P4/P5 각각에서 lo/hi 2개씩 총 6개 sub-band token + 1 enc_out = 7 memory tokens.
다중 음원의 spectral 특성이 서로 다를 때 이 7개로는 구분력이 부족하다.

### 변경 내용

#### `encoder.py` — `SLEDEncoder.forward()` 내 `_subband_pool` 함수

**변경 전 (line ~220-230):**
```python
def _subband_pool(p: torch.Tensor):
    F_dim = p.shape[2]
    mid   = F_dim // 2
    lo    = p[:, :, :mid, :].mean(dim=2).permute(0, 2, 1)
    hi    = p[:, :, mid:, :].mean(dim=2).permute(0, 2, 1)
    return lo, hi

ms_feats = []
for px in (P3, P4, P5):
    lo, hi = _subband_pool(px)
    ms_feats.extend([lo, hi])
```

**변경 후:**
```python
def _subband_pool(p: torch.Tensor, n_bands: int = 4):
    """Split frequency axis into n_bands and pool each."""
    F_dim = p.shape[2]
    bands = []
    band_size = max(1, F_dim // n_bands)
    for b in range(n_bands):
        start = b * band_size
        end = min((b + 1) * band_size, F_dim)
        if start >= F_dim:
            break
        band = p[:, :, start:end, :].mean(dim=2).permute(0, 2, 1)  # [B, T, d]
        bands.append(band)
    return bands

ms_feats = []
# P3 (highest freq resolution) → 4 bands
# P4 → 4 bands
# P5 (lowest freq resolution) → 2 bands
for px, nb in [(P3, 4), (P4, 4), (P5, 2)]:
    bands = _subband_pool(px, n_bands=nb)
    ms_feats.extend(bands)
# ms_feats: 4 + 4 + 2 = 10 sub-band tokens
```

#### `encoder.py` — `SLEDEncoder.forward()` 마지막 부분

enc_out을 추가하면 총 **11** candidate가 됨.

```python
ms_feats.append(enc_out)   # 11th candidate (was 7th)
```

---

#### `sled.py` — `SLEDv3` 클래스

**변경 전 (line 45):**
```python
N_CANDIDATES = 7   # 6 sub-band + 1 enc_out
```

**변경 후:**
```python
N_CANDIDATES = 11   # 10 sub-band + 1 enc_out
```

**변경 전 — freq_memory 구성 (line ~120):**
```python
freq_memory = torch.stack(multi_scale, dim=2).reshape(B * T, 7, d)
```

**변경 후:**
```python
freq_memory = torch.stack(multi_scale, dim=2).reshape(B * T, self.N_CANDIDATES, d)
```

---

## 추가 권장 사항 (선택)

### A. SlotDiversityLoss 가중치 증가

`train.py` line ~314:

```python
# 변경 전:
total_loss = total_loss + 0.5 * div

# 변경 후:
total_loss = total_loss + 1.0 * div
```

### B. Encoder conformer 수 증가

다중 음원의 temporal pattern 학습 강화. `sled.py` 기본값:

```python
# 변경 전:
n_conformer_layers: int = 4

# 변경 후:
n_conformer_layers: int = 6
```

### C. Spatial memory elevation 해상도 증가

`decoder.py` `SpatialBeamformingMemory.__init__()`:

```python
# 변경 전:
n_el: int = 2   # ±30° → 72 directions total

# 변경 후:
n_el: int = 4   # -45°, -15°, +15°, +45° → 144 directions total

# elevations 리스트도 수정:
elevations = [-45, -15, 15, 45]
elevations = [e * math.pi / 180.0 for e in elevations]
```

---

## 수정 후 예상 파라미터 변화

| 컴포넌트 | 변경 전 | 변경 후 | 추가 파라미터 (approx) |
|----------|---------|---------|----------------------|
| HRTFProjection | Linear(2048→256) ×2 | 2D CNN (1→32→64→128) + Linear(128→256) ×2 | ~150K |
| slot_spatial_embed | 없음 | [3, 256] | ~768 |
| N_CANDIDATES | 7 | 11 | 0 (구조 변경만) |
| conformer ×6 (선택) | 4 layers | 6 layers | ~3.2M |
| spatial_memory n_el=4 (선택) | 72 dirs | 144 dirs | ~100K |

핵심 3가지만 적용 시 ~150K 추가. 선택 사항 포함 시 ~3.5M 추가.

---

## 체크리스트

- [ ] `preprocessor.py`: ch5를 per-frame `[B, T, 64, 32]`로 변경
- [ ] `encoder.py`: `HRTFProjection`을 2D CNN 기반으로 교체
- [ ] `encoder.py`: `SLEDEncoder.forward()`의 HRTF injection을 per-frame으로 변경
- [ ] `encoder.py`: `_subband_pool`을 n_bands 파라미터 지원으로 변경
- [ ] `decoder.py`: `CrossAttentionQuerySelector`에 spatial prior 추가
- [ ] `sled.py`: `N_CANDIDATES`를 7→11로 변경, freq_memory reshape 수정
- [ ] `sled.py`: 주석 업데이트 (hrtf_ch shape)
- [ ] (선택) `train.py`: diversity loss 가중치 0.5→1.0
- [ ] (선택) conformer 수 4→6
- [ ] (선택) spatial memory elevation 2→4
