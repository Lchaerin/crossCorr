# SLED v3 Revision Plan — Multi-Source DOA Fix

## Problem
Single-source DOA works fine. 2+ sources → DOA degrades sharply.

## Root Causes
1. **No slot diversity enforcement** — slots collapse to same direction
2. **Memory lacks spatial structure** — 7 tokens are frequency-only, no directional info
3. **No inter-slot competition in decoder** — slots refine independently

## File Map
```
model/
  preprocessor.py  — AudioPreprocessor (stereo → [B,5,64,T] + hrtf [B,64,32])
  encoder.py       — SLEDEncoder (ConvBlocks → BiFPN → Conformer → 7 multi-scale feats)
  decoder.py       — CrossAttentionQuerySelector, ContrastiveDeNoising, DecoderLayer, IterativeRefinementDecoder
  heads.py         — DetectionHeads (class_logits, doa_vec, loudness, confidence)
  sled.py          — SLEDv3 (orchestrates all above)
train.py           — training loop + loss
```

## Changes (apply in order)

### 1. Add SlotDiversityLoss — new file `losses.py`

```python
class SlotDiversityLoss(nn.Module):
    """Repulsion between active slot DOA predictions."""
    def forward(self, doa_vecs: Tensor, mask: Tensor) -> Tensor:
        # doa_vecs: [B,T,S,3], mask: [B,T,S] bool
        # For each active slot pair (i,j), penalize high cosine similarity
        S = doa_vecs.shape[2]
        loss = 0.0
        count = 0
        for i in range(S):
            for j in range(i+1, S):
                both = mask[:,:,i] & mask[:,:,j]
                if both.any():
                    cos = F.cosine_similarity(doa_vecs[:,:,i], doa_vecs[:,:,j], dim=-1)
                    loss += cos[both].clamp(min=0).mean()
                    count += 1
        return loss / max(count, 1)
```

Add to training loss: `total_loss += 0.5 * diversity_loss(pred_doa, gt_mask)`

### 2. Add SpatialBeamformingMemory — in `decoder.py`

New class that creates direction-aware memory tokens from a fixed angular grid.

```python
class SpatialBeamformingMemory(nn.Module):
    """Fixed angular grid → spatial tokens that fuse with encoder output."""
    def __init__(self, d_model=256, n_az=36, n_el=2, n_heads=8):
        # n_az * n_el = 72 spatial tokens
        # direction_grid: [72, 3] unit vectors on sphere
        # spatial_proj: Linear(3 → d_model) + GELU + Linear(d_model → d_model)
        # fusion_attn: MHA(d_model, n_heads) — spatial queries attend to enc_out
        # norm: LayerNorm(d_model)

    def forward(self, enc_out: Tensor) -> Tensor:
        # enc_out: [B, T, d]
        # Returns: [B*T, 72, d]
        # 1. Project direction_grid → [72, d]
        # 2. Expand to [B*T, 72, d]
        # 3. Cross-attend with enc_out (broadcast per-frame)
        # 4. Residual + norm
```

### 3. Modify decoder to dual-memory cross-attention — in `decoder.py`

Replace `DecoderLayer` with `SlotCompetitionLayer`:

```python
class SlotCompetitionLayer(nn.Module):
    """self-attn → spatial cross-attn → freq cross-attn → FFN"""
    def __init__(self, d_model=256, n_heads=8, ffn_dim=1024, dropout=0.1):
        # self_attn + norm1
        # spatial_cross_attn + norm2  (Q=slots, KV=spatial_memory)
        # freq_cross_attn + norm3     (Q=slots, KV=freq_memory [7 tokens])
        # ffn + norm4

    def forward(self, slots, freq_memory, spatial_memory, self_attn_mask=None):
        # Returns: (slots, spatial_attn_weights)
        # spatial_attn_weights: [B*T, S, 72] — used for diversity regularization
```

Update `IterativeRefinementDecoder`:
- Accept `spatial_memory` as additional input
- Pass both memories to each layer
- DOA refinement injection stays the same
- Signature: `forward(queries, freq_memory, spatial_memory, attn_mask) -> list`

### 4. Update SLEDv3.forward — in `sled.py`

```python
# After encoder:
spatial_memory = self.spatial_memory(enc_out)  # [B*T, 72, d]

# Decoder call changes:
all_layer_outputs = self.decoder(all_queries, freq_memory, spatial_memory, attn_mask)
```

Add `self.spatial_memory = SpatialBeamformingMemory(d_model)` in `__init__`.

### 5. Scale up model (optional, after verifying improvement)

- `d_model`: 256 → 384 or 512
- `n_decoder_layers`: 4 → 6
- `ffn_dim` in decoder: d_model * 4
- `n_conformer_layers`: 4 → 6
- `n_slots`: 3 → 5 (extra slots + confidence threshold for dynamic count)

### 6. Training loss weight adjustment — in `train.py`

- Increase DOA loss weight relative to classification (e.g. 2x–3x)
- Add diversity loss (λ=0.5)
- Optional: add spatial attention entropy regularization to prevent all slots attending same directions

## Validation
- Compare on 1-source, 2-source, 3-source separately
- Key metric: DOA error (degrees) per source count
- Expect: 1-source stays same, 2+ source improves significantly
