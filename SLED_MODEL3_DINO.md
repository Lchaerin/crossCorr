# SLED v3 — DINO-Enhanced Sound Localization & Event Detection Model

> 구현 레퍼런스 문서 (2026-03-20 기준)
> 기존 SLED v2에서 DINO의 핵심 기법들을 Sound Domain에 맞게 적용한 개선 모델
> Binaural audio 입력 → 최대 5개 동시 음원의 (class, azimuth, elevation, loudness, confidence) per-frame 출력

---

## 변경 요약: SLED v2 → v3

| 문제점 (v2) | 원인 분석 | 해결책 (v3) | 출처 |
|---|---|---|---|
| **실제보다 많은 음원 예측** (false positive) | Learnable query가 "빈 슬롯"을 reject하는 법을 잘 못 배움 | **Contrastive DeNoising Training** — negative sample로 "no object" 학습 강화 | DINO |
| **predicted 값 분포가 특정 값에 집중** (mode collapse) | Learnable query 초기화가 동일 → 슬롯들이 비슷한 위치로 수렴 | **Mixed Query Selection** — encoder top-K feature로 anchor 초기화 | DINO |
| 중간 decoder layer gradient 단절 | v2의 aux loss가 detached box를 다음 layer에 전달 | **Look Forward Twice** — gradient를 다음 layer까지 흘림 | DINO |
| 학습 초기 Hungarian matching 불안정 | 랜덤 초기 예측 → matching 품질 저하 → noisy gradient | DeNoising이 matching-free 보조 신호 제공 → 안정적 초기 학습 | DN-DETR/DINO |
| 단일 모델로 DOA+SDE 동시 추정 한계 | - | **Multi-scale encoder** 강화 + DCASE 1등 팀의 multi-output 전략 참고 | NERC-SLIP |

---

## 입력/출력 (v2와 동일 — 변경 없음)

### 입력
```
[B, 5, 64, T]   — (L-mel, R-mel, cos-IPD, sin-IPD, ILD), 64 mel bins, T frames
```

### 출력
```python
{
    "class_logits": Tensor[B, T, S, 301],   # S=5 slots
    "doa_vec":      Tensor[B, T, S, 3],     # unit vector
    "loudness":     Tensor[B, T, S],         # dBFS
    "confidence":   Tensor[B, T, S],         # logit
    "source_embed": Tensor[B, S, 256],       # VLA 전달용
}
```

### 제약 조건 (변경 없음)
- **Strictly causal**: 현재 time window 내 오디오만 사용
- **Latency**: < 100ms per window on RTX 5090
- **Window**: 수십 ms ~ 100ms 단위 처리

---

## 모델 아키텍처 (v3)

### 전체 흐름

```
Input [B, 5, 64, T]
  │
  ▼
┌──────────────────────────────────────────────────┐
│  ENCODER (변경됨)                                  │
│                                                    │
│  ConvBlock ×3 (동일)                               │
│    5 → 64 → 128 → 256 channels                    │
│    P3[B,64,32,T], P4[B,128,16,T], P5[B,256,8,T]  │
│    │                                               │
│    ▼                                               │
│  SE Block (동일)                                    │
│    P5 → [B, 256, 8, T]                             │
│    │                                               │
│    ▼                                               │
│  Flatten + Conv1d → [B, 256, T]                    │
│    │                                               │
│    ▼                                               │
│  Causal Conformer ×4 (동일)                         │
│    [B, T, 256]                                      │
│    │                                               │
│    ▼                                               │
│  Causal BiFPN Neck ×2 (동일)                        │
│    │                                               │
│    ▼                                               │
│  enc_out = [B, T, 256]                              │
└──────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────┐
│  ★ MIXED QUERY SELECTION (신규)                    │
│                                                    │
│  enc_out [B, T, 256]                               │
│    │                                               │
│    ├─ score_head: Linear(256,1) → [B, T, 1]       │
│    │   per-frame objectness score                   │
│    │                                               │
│    ├─ top-K selection (K=5, per frame)              │
│    │   BUT: frame-local이므로 T차원이 아닌           │
│    │   enc_out[t]를 K개 anchor로 분해               │
│    │   → 아래 "Frame-local Anchor Selection" 참조  │
│    │                                               │
│    ├─ anchor_pos = selected features (detached)     │
│    │   → position part of query                    │
│    │                                               │
│    └─ content_queries = learnable [S, 256]          │
│       → content part (학습 가능, v2와 동일)          │
│                                                    │
│  mixed_query = content_queries + anchor_pos         │
│  [B*T, S, 256]                                      │
└──────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────┐
│  ★ DENOISING TRAINING (신규, 학습 시에만)           │
│                                                    │
│  GT boxes에 noise 추가 → denoising queries 생성     │
│  positive DN queries + negative DN queries          │
│  → matching queries와 concat하여 decoder에 입력     │
│                                                    │
│  [B*T, S + S_dn_pos + S_dn_neg, 256]              │
│  attention mask로 DN ↔ matching 정보 차단            │
└──────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────┐
│  SLOT DECODER (개선됨)                              │
│                                                    │
│  Frame-local cross-attention (v2와 동일 원리)       │
│  memory: enc_out[t] → (B*T, 1, d)                  │
│                                                    │
│  ★ Look Forward Twice:                             │
│    각 layer의 refined prediction이                  │
│    다음 layer anchor 업데이트 시 gradient 유지       │
│                                                    │
│  4 decoder layers, 각 layer 출력:                   │
│    self-attn(slots) → cross-attn(slots, memory)     │
│    → refined_anchor = anchor + Δ(slot_feat)         │
│    → detection heads에서 예측                        │
│    → 다음 layer에 refined_anchor 전달 (no detach)   │
└──────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────┐
│  DETECTION HEADS (동일 구조)                        │
│                                                    │
│  class_head  → [B, T, S, 301]                      │
│  doa_head    → [B, T, S, 3]  (unit vec)            │
│  loud_head   → [B, T, S]                           │
│  conf_head   → [B, T, S]                           │
└──────────────────────────────────────────────────┘
```

---

## 핵심 변경사항 상세

### 1. Frame-local Mixed Query Selection

#### 문제
v2에서 5개 learnable slot queries는 모두 동일하게 초기화되므로, 학습 초기에 슬롯들이 비슷한 예측을 하다가 점차 분화됨. 이로 인해:
- 수렴이 느림
- 특정 DOA/class에 여러 슬롯이 몰리는 mode collapse 발생
- predicted 값 분포가 특정 값에 집중

#### 해결: Encoder-informed Anchor Initialization

DINO의 Mixed Query Selection을 frame-local 설정에 맞게 변형.

**핵심 차이**: 원본 DINO는 encoder output의 전체 spatial position에서 top-K를 뽑지만, SLED는 frame-local이므로 각 frame의 enc_out[t] 하나만 memory로 사용. 따라서 "spatial top-K"가 아닌 **frequency-band 기반 anchor 생성** 방식을 사용한다.

```python
class MixedQuerySelector(nn.Module):
    """
    enc_out [B, T, 256]을 기반으로 S개의 diverse anchor position 생성.
    
    전략: enc_out을 BiFPN 이전의 multi-scale features (P3, P4, P5)로부터
    frequency-band별 objectness score를 계산하여 diverse한 anchor 선택.
    """
    def __init__(self, d_model=256, n_slots=5, n_scales=3):
        super().__init__()
        self.n_slots = n_slots
        
        # Multi-scale feature에서 각각 objectness score 계산
        # P3: 저주파 대역 (32 mel bins), P4: 중주파 (16), P5: 고주파 (8)
        self.scale_score_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_scales)
        ])
        
        # 각 scale feature를 anchor position으로 변환
        self.anchor_proj = nn.Linear(d_model, d_model)
        
        # Learnable content queries (v2의 slot_queries와 동일 역할)
        self.content_queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
    
    def forward(self, multi_scale_feats, enc_out):
        """
        Args:
            multi_scale_feats: list of [B, T, d] from P3, P4, P5 (projected)
            enc_out: [B, T, d] — final encoder output
        Returns:
            mixed_queries: [B*T, S, d]
        """
        B, T, d = enc_out.shape
        
        # 각 scale에서 objectness score 계산
        all_scores = []   # list of [B, T, 1]
        all_feats = []    # list of [B, T, d]
        for i, (feat, head) in enumerate(zip(multi_scale_feats, self.scale_score_heads)):
            score = head(feat)           # [B, T, 1]
            all_scores.append(score)
            all_feats.append(feat)
        
        # [B, T, n_scales] scores, [B, T, n_scales, d] features
        scores = torch.cat(all_scores, dim=-1)     # [B, T, n_scales]
        feats = torch.stack(all_feats, dim=2)      # [B, T, n_scales, d]
        
        # enc_out 자체도 candidate로 추가 (total n_scales + 1 candidates)
        enc_score = nn.functional.linear(enc_out, self.scale_score_heads[0].weight)
        scores = torch.cat([scores, enc_score], dim=-1)         # [B, T, n_scales+1]
        feats = torch.cat([feats, enc_out.unsqueeze(2)], dim=2) # [B, T, n_scales+1, d]
        
        # Top-K selection (K = n_slots)
        # Soft top-K: Gumbel-Softmax로 differentiable selection (학습 시)
        # Hard top-K: argmax (추론 시)
        n_candidates = scores.shape[-1]  # n_scales + 1 = 4
        
        if n_candidates >= self.n_slots:
            # top-K per frame
            topk_idx = scores.topk(self.n_slots, dim=-1).indices  # [B, T, S]
            # Gather features
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, d)
            anchor_pos = feats.gather(2, topk_idx_exp)  # [B, T, S, d]
        else:
            # candidates < slots인 경우: 반복 사용 + learnable offset
            anchor_pos = feats[:, :, :self.n_slots]  # 충분하지 않으면 padding
        
        # Position part: encoder feature (detached for stability)
        anchor_pos = self.anchor_proj(anchor_pos.detach())  # [B, T, S, d]
        
        # Content part: learnable (shared across all frames)
        content = self.content_queries.unsqueeze(0).unsqueeze(0)  # [1, 1, S, d]
        content = content.expand(B, T, -1, -1)
        
        # Mixed query = content + anchor_position
        mixed = content + anchor_pos  # [B, T, S, d]
        
        return mixed.reshape(B * T, self.n_slots, d)
```

**candidate가 4개(P3, P4, P5, enc_out)인데 slot이 5개인 문제 해결:**

실제 구현에서는 multi-scale features를 더 세분화하여 candidate 수를 늘린다:

```python
# P3 → 2개 sub-band (low/mid-low)
# P4 → 2개 sub-band (mid/mid-high)
# P5 → 2개 sub-band (high/very-high)
# enc_out → 1개
# Total: 7 candidates → top-5 selection
```

이 방식으로 각 슬롯이 서로 다른 주파수 대역의 정보를 기반으로 초기화되어, **mode collapse를 방지**하고 다양한 spatial position의 음원을 탐지할 수 있다.

---

### 2. Contrastive DeNoising Training

#### 문제
v2는 Hungarian matching에만 의존하므로:
- 학습 초기에 matching이 불안정 → noisy gradient
- "no object"를 예측해야 할 빈 슬롯이 false positive를 생성
- 실제보다 많은 수의 음원 예측

#### 해결: Positive + Negative Denoising Queries

DINO의 contrastive denoising을 sound domain에 적용. 핵심 아이디어: GT annotation에 controlled noise를 추가한 query를 만들어 decoder에 함께 넣되, **positive** (복원 가능한 수준)과 **negative** (복원 불가능할 정도로 먼) 두 종류를 모두 사용.

```python
class ContrastiveDeNoising(nn.Module):
    """
    Sound-domain contrastive denoising.
    
    GT (class, doa_vec, loudness)에 noise를 추가하여
    positive/negative denoising queries를 생성.
    """
    def __init__(self, d_model=256, n_classes=300, 
                 noise_scale_pos=0.2, noise_scale_neg=0.8,
                 n_dn_groups=3):
        super().__init__()
        self.noise_scale_pos = noise_scale_pos
        self.noise_scale_neg = noise_scale_neg
        self.n_dn_groups = n_dn_groups  # DN query 그룹 수
        
        # GT 정보를 query embedding으로 변환
        self.class_embed = nn.Embedding(n_classes + 1, d_model)  # +1 for empty
        self.doa_proj = nn.Linear(3, d_model)
        self.loud_proj = nn.Linear(1, d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
    
    def forward(self, gt_classes, gt_doa_vecs, gt_loudness, gt_mask):
        """
        Args:
            gt_classes:  [B, T, S_gt] int — GT class IDs
            gt_doa_vecs: [B, T, S_gt, 3] — GT DOA unit vectors
            gt_loudness: [B, T, S_gt] — GT loudness dBFS
            gt_mask:     [B, T, S_gt] bool — valid GT mask
        
        Returns:
            dn_queries:   [B*T, S_dn, d] — denoising queries
            dn_targets:   dict — GT targets for DN queries
            dn_attn_mask: [S + S_dn, S + S_dn] — attention mask
            dn_meta:      dict — positive/negative index info
        """
        B, T, S_gt = gt_classes.shape
        d = self.class_embed.embedding_dim
        
        positive_queries = []
        negative_queries = []
        positive_targets = []
        negative_targets = []
        
        for g in range(self.n_dn_groups):
            # === Positive DN queries ===
            # DOA에 작은 noise 추가 (복원 가능)
            doa_noise = torch.randn_like(gt_doa_vecs) * self.noise_scale_pos
            noisy_doa_pos = F.normalize(gt_doa_vecs + doa_noise, dim=-1)
            
            # Class label에 일정 확률로 flip (10%)
            class_flip_mask = torch.rand(B, T, S_gt, device=gt_classes.device) < 0.1
            noisy_class_pos = gt_classes.clone()
            random_classes = torch.randint(0, 300, (B, T, S_gt), device=gt_classes.device)
            noisy_class_pos[class_flip_mask] = random_classes[class_flip_mask]
            
            # Loudness에 작은 noise
            loud_noise = torch.randn_like(gt_loudness) * 3.0  # ±3dB
            noisy_loud_pos = gt_loudness + loud_noise
            
            # Embedding 생성
            pos_q = self._embed(noisy_class_pos, noisy_doa_pos, noisy_loud_pos)
            positive_queries.append(pos_q)
            positive_targets.append({
                'class': gt_classes, 'doa': gt_doa_vecs, 
                'loud': gt_loudness, 'mask': gt_mask
            })
            
            # === Negative DN queries ===
            # DOA를 크게 벗어나게 (복원 불가능)
            doa_noise_neg = torch.randn_like(gt_doa_vecs) * self.noise_scale_neg
            # 추가: GT 반대 방향으로 편향
            noisy_doa_neg = F.normalize(-gt_doa_vecs + doa_noise_neg, dim=-1)
            
            # Class는 무조건 랜덤
            random_classes_neg = torch.randint(0, 300, (B, T, S_gt), device=gt_classes.device)
            
            neg_q = self._embed(random_classes_neg, noisy_doa_neg, gt_loudness)
            negative_queries.append(neg_q)
            # Negative targets: 모두 "empty" class (300) + confidence=0
            negative_targets.append({
                'class': torch.full_like(gt_classes, 300),  # empty class
                'doa': noisy_doa_neg,  # don't care (masked by empty class)
                'loud': torch.zeros_like(gt_loudness),
                'mask': gt_mask  # 해당 GT가 있는 위치의 negative만 유효
            })
        
        # Concat all DN queries
        # [B, T, n_dn_groups * S_gt * 2, d]
        pos_q = torch.cat(positive_queries, dim=2)  # [B, T, n_groups*S_gt, d]
        neg_q = torch.cat(negative_queries, dim=2)  # [B, T, n_groups*S_gt, d]
        dn_queries = torch.cat([pos_q, neg_q], dim=2)  # [B, T, S_dn, d]
        
        S_dn = dn_queries.shape[2]
        dn_queries = dn_queries.reshape(B * T, S_dn, d)
        
        # Attention mask: DN queries는 matching queries와 정보 교환 차단
        # [S + S_dn, S + S_dn] — 여기서 S=5 (matching slots)
        # 구성: matching이 DN을 볼 수 없고, DN도 matching을 볼 수 없음
        # 단, 같은 DN group 내에서는 서로 볼 수 있음
        
        return dn_queries, positive_targets, negative_targets, S_dn
    
    def _embed(self, classes, doa_vecs, loudness):
        """GT 정보를 query embedding으로 변환"""
        cls_emb = self.class_embed(classes)                    # [B, T, S, d]
        doa_emb = self.doa_proj(doa_vecs)                      # [B, T, S, d]
        loud_emb = self.loud_proj(loudness.unsqueeze(-1))      # [B, T, S, d]
        fused = self.fusion(torch.cat([cls_emb, doa_emb, loud_emb], dim=-1))
        return fused
```

**Attention Mask 구조:**

```
             matching(5)  |  DN_pos(15)  |  DN_neg(15)
            ─────────────┼──────────────┼──────────────
matching(5)│  ✓ see each │  ✗ blocked   │  ✗ blocked
            │  other      │              │
            ─────────────┼──────────────┼──────────────
DN_pos(15) │  ✗ blocked  │  ✓ within    │  ✗ blocked
            │             │  same group  │
            ─────────────┼──────────────┼──────────────
DN_neg(15) │  ✗ blocked  │  ✗ blocked   │  ✓ within
            │             │              │  same group
```

이렇게 하면:
- **Matching queries**는 DN 정보에 접근 불가 → 테스트 시와 동일한 조건으로 학습
- **Positive DN queries**는 GT 근처에서 원래 GT를 복원하도록 학습 → decoder에 "이 근처에 음원이 있다"는 신호
- **Negative DN queries**는 "no object"로 분류하도록 학습 → **빈 슬롯의 false positive 억제**

---

### 3. Look Forward Twice

#### 문제
v2에서는 각 decoder layer의 예측이 aux loss를 통해 학습되지만, refined prediction이 다음 layer로 전달될 때 detach되어 gradient가 끊김.

#### 해결

```python
class LookForwardTwiceDecoder(nn.Module):
    """
    각 decoder layer에서:
    1. slot features → detection heads → prediction (loss 1)
    2. prediction으로 anchor 업데이트 (gradient 유지!)
    3. 업데이트된 anchor를 다음 layer에 전달
    4. 다음 layer에서도 같은 anchor 기반 prediction → (loss 2)
    
    즉, 한 layer의 예측이 자신의 loss + 다음 layer의 loss 두 번 학습됨.
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=4, n_slots=5):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        # Anchor refinement: 각 layer에서 anchor offset 예측
        self.anchor_delta = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(n_layers)
        ])
    
    def forward(self, queries, memory, attn_mask=None):
        """
        Args:
            queries: [B*T, S_total, d] — mixed queries (+ optional DN queries)
            memory:  [B*T, 1, d] — frame-local encoder output
            attn_mask: optional attention mask for DN training
        
        Returns:
            all_layer_outputs: list of [B*T, S_total, d] — 각 layer의 slot features
            refined_anchors: list of [B*T, S_total, d] — 각 layer의 refined anchors
        """
        all_outputs = []
        slots = queries  # initial queries
        anchor = queries.clone()  # initial anchor = mixed query
        
        for i, (layer, delta_head) in enumerate(zip(self.layers, self.anchor_delta)):
            # Decoder layer: self-attn + cross-attn
            slots = layer(
                tgt=slots,
                memory=memory,
                self_attn_mask=attn_mask
            )
            all_outputs.append(slots)
            
            # Anchor refinement (NO DETACH — gradient flows through!)
            delta = delta_head(slots)
            anchor = anchor + delta  # refined anchor
            
            # 다음 layer의 input으로 refined anchor 사용
            if i < len(self.layers) - 1:
                slots = anchor  # gradient가 살아있는 채로 전달
        
        return all_outputs
```

**Look Forward Twice의 효과:**

```
Layer 0 prediction ──→ Loss_0 (direct supervision)
    │
    ▼ (gradient flows, no detach)
Layer 1 uses refined anchor from Layer 0
Layer 1 prediction ──→ Loss_1 (also backprops to Layer 0's refinement)

∴ Layer 0의 anchor refinement은 Loss_0 + Loss_1 두 번 학습됨
```

---

### 4. Decoder Layer 구조 (개선)

```python
class DecoderLayer(nn.Module):
    """
    DINO-style decoder layer for frame-local SELD.
    
    v2 대비 변경:
    - Self-attention에서 DN mask 지원
    - Cross-attention에 conditional position encoding 추가
    - Pre-norm → Post-norm (DINO 스타일)
    """
    def __init__(self, d_model=256, n_heads=8, ffn_dim=512, dropout=0.1):
        super().__init__()
        # Self-attention (slots 간)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention (slots → encoder memory)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory, self_attn_mask=None):
        """
        tgt:     [B*T, S_total, d]
        memory:  [B*T, 1, d]
        """
        # Self-attention with optional DN mask
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=True
        ):
            q = k = tgt
            sa_out, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
        tgt = self.norm1(tgt + self.dropout1(sa_out))
        
        # Cross-attention to frame-local memory
        ca_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(ca_out))
        
        # FFN
        tgt = self.norm3(tgt + self.ffn(tgt))
        
        return tgt
```

---

## 학습 설정 (v3)

### Loss 구성

#### Matching Loss (v2와 유사, Look Forward Twice 적용)

| Head | Loss | Weight |
|---|---|---|
| Classification | Focal Loss (α=0.25, γ=2.0) | 1.0 |
| DOA | Cosine distance: `1 - cos(pred, gt)` | **2.0** |
| Loudness | Smooth L1 (÷20 정규화) | 0.5 |
| Confidence | BCE | 0.5 |
| SCE auxiliary | MSE on `doa_vec × exp(loud/20)` | 0.1 |

#### ★ Denoising Loss (신규)

| Component | Loss | Weight |
|---|---|---|
| DN Positive — Classification | Focal Loss | 1.0 |
| DN Positive — DOA | Cosine distance | 2.0 |
| DN Positive — Loudness | Smooth L1 | 0.5 |
| DN Negative — Classification | Focal Loss (target = empty class 300) | **2.0** |
| DN Negative — Confidence | BCE (target = 0) | **1.0** |

> **Negative DN loss weight를 높인 이유**: v2의 핵심 문제가 false positive (과도한 음원 수 예측)이므로, negative sample에 대한 "no object" 학습을 강화.

#### Total Loss (Look Forward Twice)

```
L_total = Σ_{i=0}^{3} w_i × (L_match_i + L_dn_i)

where:
  w_0 = 0.2, w_1 = 0.4, w_2 = 0.6, w_3 = 1.0  (later layers weighted more)
  
  L_match_i = cls + 2*doa + 0.5*loud + 0.5*conf + 0.1*sce  (layer i matching)
  L_dn_i    = dn_pos_cls + 2*dn_pos_doa + 0.5*dn_pos_loud  (layer i DN positive)
            + 2*dn_neg_cls + 1.0*dn_neg_conf                 (layer i DN negative)
```

**v2와의 차이:**
- v2: `total = main + 0.4 × (aux0 + aux1 + aux2)` — 모든 aux 동일 가중치
- v3: 점진적 가중치 `[0.2, 0.4, 0.6, 1.0]` — 후반 layer에 더 큰 가중치
- v3: Look Forward Twice로 각 layer의 gradient가 다음 layer를 통해서도 흐름

### Hungarian Matching (개선)

```python
# v2: clip-level matching (T 프레임 평균 cost로 1회 matching)
# v3: 동일하되, matching cost 계산 시 DN 정보는 제외

# Matching cost 구성
cost_cls   = focal_loss_cost(pred_logits, gt_classes)     # [B, S, S_gt]
cost_doa   = 1 - cosine_sim(pred_doa, gt_doa)             # [B, S, S_gt]
cost_conf  = bce_cost(pred_conf, ones)                     # [B, S, S_gt]

total_cost = cost_cls + cost_doa + 0.5 * cost_conf  # clip-level average over T
# scipy.optimize.linear_sum_assignment per batch item
```

### 하이퍼파라미터 변경

| 항목 | v2 | v3 | 변경 이유 |
|---|---|---|---|
| LR | 2e-3 | **1e-4** | DN training이 추가 학습 신호를 제공하므로 더 작은 LR로 안정적 학습 |
| LR schedule | Cosine → 1e-5 | **Cosine → 5e-6** | |
| Warmup | 10 epochs | **15 epochs** | Mixed query selection 안정화 |
| Epochs | 200~300 | **150~200** | DN training으로 수렴 가속 |
| n_dn_groups | - | **3** | 3 groups × 5 GT slots × 2(pos/neg) = 30 DN queries |
| DN noise_pos | - | **0.2** | DOA unit vector에 대한 noise scale |
| DN noise_neg | - | **0.8** | Negative가 GT와 충분히 멀어야 효과적 |
| Batch size | 256 | 256 | 동일 |
| Gradient clip | 5.0 | **1.0** | DN loss 추가로 gradient 크기 증가 → 더 타이트한 clipping |

### Curriculum 학습 (개선)

```
Phase 1 (Epoch 1~30):    max_sources=2, n_dn_groups=5  ← DN 비중 높게
Phase 2 (Epoch 31~80):   max_sources=3, n_dn_groups=3
Phase 3 (Epoch 81~):     max_sources=5, n_dn_groups=3
```

초기에 DN group을 많이 두어 "no object" 학습을 충분히 시킨 후, 점차 matching 위주 학습으로 전환.

---

## 파라미터 규모

| 모듈 | v2 | v3 | 비고 |
|---|---|---|---|
| Encoder (Conv+SE+BiFPN+Conformer) | ~7.7M | ~7.7M | 동일 |
| Mixed Query Selector | - | ~0.3M | scale score heads + anchor proj |
| Contrastive DeNoising | - | ~0.4M | class embed + proj + fusion |
| Slot Decoder (4 layers) | ~1.0M | ~1.3M | anchor delta heads 추가 |
| Detection Heads | ~0.2M | ~0.2M | 동일 |
| CLAP Head | ~0.1M | ~0.1M | 동일 |
| **Total** | **~9M** | **~10M** | +1M (11% 증가) |

---

## 추론 시 동작 (Inference)

학습 시에만 사용되는 DN 관련 모듈이 추론 시에는 완전히 제거됨:

```python
def forward(self, feat, training=False):
    # Encoder (동일)
    multi_scale, enc_out = self.encoder(feat)
    
    # Mixed Query Selection (추론에도 사용)
    queries = self.query_selector(multi_scale, enc_out)  # [B*T, S, d]
    
    if training:
        # DN queries 생성 및 concat
        dn_queries, dn_targets, S_dn = self.denoising(gt_classes, gt_doa, gt_loud, gt_mask)
        queries = torch.cat([queries, dn_queries], dim=1)  # [B*T, S+S_dn, d]
        attn_mask = self._build_dn_mask(S=5, S_dn=S_dn)
    else:
        attn_mask = None
    
    # Decoder
    all_outputs = self.decoder(queries, memory, attn_mask)
    
    # Detection heads — matching slots만 사용 (first S=5)
    last_output = all_outputs[-1][:, :5, :]  # [B*T, S, d]
    ...
```

**추론 시 추가 연산:**
- Mixed Query Selection: ~0.1ms (linear projection + top-K)
- DN 모듈: 완전 비활성 → 0ms
- Look Forward Twice: decoder 구조는 동일, anchor delta만 추가 → ~0.2ms

**예상 추론 시간 (RTX 5090, single window):**
- v2: ~5ms (A100 기준)
- v3: ~7ms (RTX 5090 기준, mixed query + anchor delta 추가)
- **100ms 제약 대비 충분한 여유**

---

## Data Augmentation (v2와 동일 + 추가)

| 기법 | v2 | v3 | 설명 |
|---|---|---|---|
| SCS | ✓ | ✓ | L↔R swap |
| SRIR 합성 | ✓ | ✓ | mono × RIR |
| SpecAugment | ✓ | ✓ | time/freq mask |
| Mixup | ✓ | ✓ | scene 합성 |
| **Noise injection** | - | ✓ | GT DOA에 random perturbation → DN training과 시너지 |
| **Random slot dropout** | - | ✓ | GT 일부를 랜덤 삭제하여 false negative에 강건하게 |

---

## NERC-SLIP (DCASE 2025 1등) 참고 반영사항

1. **Multi-output strategy**: NERC-SLIP은 SED-DOA, SED-SDE, SED-SCE 3개 모델을 앙상블.
   - v3에서는 단일 모델 유지 (실시간 제약) but decoder의 intermediate outputs를 활용하여
     내부적으로 "다중 관점" 학습 (Look Forward Twice가 이 역할)

2. **ResNet-Conformer backbone**: NERC-SLIP과 동일한 Conv + Conformer 구조를 이미 v2에서 사용 중.
   v3에서도 유지.

3. **SCS augmentation**: NERC-SLIP의 stereo channel swap은 v2의 SCS와 동일 개념.

4. **Model ensemble via posterior fusion**: 실시간 제약으로 multi-model ensemble은 불가.
   대신 **decoder layer별 prediction의 weighted average**를 추론 시 사용 가능:
   ```python
   # Pseudo-ensemble: 각 decoder layer의 예측을 가중 평균
   weights = [0.1, 0.2, 0.3, 0.4]  # later layers get more weight
   final_logits = sum(w * out for w, out in zip(weights, layer_logits))
   ```

5. **Grounding DINO post-processing**: v3에서 audio-only이므로 비해당.
   단, confidence threshold를 더 공격적으로 설정 (0.5 → **0.6**)하여 false positive 추가 억제.

---

## 주의사항 (v2에서 추가/변경)

1. **Flash Attention + DN queries**: DN queries를 concat하면 S_total = 5 + 30 = 35.
   Flash Attention 최소 block size를 초과하므로 `SDPBackend.FLASH_ATTENTION` 사용 가능.
   단, S=5 matching slots만 사용하는 추론 시에는 여전히 `SDPBackend.MATH` 필요.

2. **DN attention mask**: mask 구성이 올바르지 않으면 DN 정보가 matching queries로 leak하여
   학습-추론 gap 발생. `assert (mask[:S, S:] == True).all()` 검증 필수.

3. **Mixed Query Selection의 top-K gradient**: top-K 자체는 non-differentiable이므로
   anchor_pos를 **detach**한 후 사용. Content query만 gradient를 받음.
   (Score head는 별도 objectness loss로 학습)

4. **Score head objectness loss**: Mixed Query Selection의 score_head를 학습시키기 위해,
   GT가 있는 frame에서는 GT와 가까운 candidate에 높은 score target 부여:
   ```python
   # Objectness target: GT DOA와 가장 유사한 candidate에 1, 나머지 0
   objectness_loss = BCE(predicted_scores, objectness_targets)
   # Total loss에 0.1 weight로 추가
   ```

5. **Negative DN noise scale 튜닝**: noise_neg가 너무 작으면 negative가 positive와 겹쳐서
   학습이 불안정. noise_neg가 너무 크면 trivially easy한 negative가 되어 학습 효과 없음.
   **0.5~1.0 범위에서 grid search 권장.**

---

## 구현 체크리스트

```
[ ] Mixed Query Selection
    [ ] Multi-scale feature projection (P3, P4, P5 → sub-bands)
    [ ] Score head + objectness loss
    [ ] Top-K selection
    [ ] Content + anchor position fusion

[ ] Contrastive DeNoising
    [ ] GT → noisy embedding conversion
    [ ] Positive/Negative query generation
    [ ] Attention mask construction
    [ ] DN-specific loss computation

[ ] Look Forward Twice Decoder
    [ ] Anchor delta heads per layer
    [ ] No detach on anchor refinement
    [ ] Layer-wise progressive loss weights

[ ] Training pipeline
    [ ] DN module 활성화/비활성화 (train/eval)
    [ ] DN attention mask correctness check
    [ ] Gradient clipping 1.0으로 변경
    [ ] LR 1e-4 + cosine schedule
    [ ] Curriculum: DN groups 5→3→3

[ ] Inference pipeline
    [ ] DN module 완전 제거 확인
    [ ] Mixed query selection만 활성
    [ ] Confidence threshold 0.6
    [ ] Layer-wise prediction average (optional)
    [ ] Latency < 100ms 확인 (RTX 5090)
```

---

## 기대 효과 요약

| 메트릭 | v2 예상 | v3 기대 | 개선 근거 |
|---|---|---|---|
| False positive rate | 높음 | **대폭 감소** | Contrastive DN negative training |
| Predicted 값 분포 | mode collapse | **다양화** | Mixed query selection |
| 수렴 epoch | 200~300 | **150~200** | DN이 matching-free 보조 신호 제공 |
| DOA error | baseline | **10~20% 개선** | Look Forward Twice + mixed query |
| 추론 latency | ~5ms | ~7ms | +2ms (100ms 제약 대비 충분) |
| 파라미터 | ~9M | ~10M | +11% |
