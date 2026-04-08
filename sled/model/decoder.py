#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Decoder
==================
Components
----------
  CrossAttentionQuerySelector
      Fully differentiable replacement for MixedQuerySelector.
      Learnable slot queries attend to all K=7 candidates via cross-attention.
      No topk, no detach → scorer receives real gradients.

  ContrastiveDeNoising
      Generates DN queries from GT labels (training only).
      Fixed: per-frame class shuffle for negatives (was all-frames same perm).

  SpatialBeamformingMemory
      Creates 72 direction-aware memory tokens from a fixed azimuth×elevation
      grid.  Spatial queries cross-attend to the encoder output, producing
      direction-specific features that give the decoder explicit spatial context.

  SlotCompetitionLayer
      self-attn → spatial cross-attn → freq cross-attn → FFN (post-norm).
      Replaces DecoderLayer.  Slots first attend to the spatial-memory tokens
      (direction grid) then to the 7 frequency-scale tokens from the encoder.
      Inter-slot self-attention creates competition so slots cannot all collapse
      to the same direction.

  IterativeRefinementDecoder
      Replaces LookForwardTwiceDecoder.
      After each layer predicts a rough DOA unit vector from matching queries
      and injects it as a learned positional embedding into the next layer,
      giving true iterative geometric refinement.
      Now accepts both freq_memory (11 tokens) and spatial_memory (72 tokens).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CrossAttentionQuerySelector
# =============================================================================

class CrossAttentionQuerySelector(nn.Module):
    """Selects queries via cross-attention over K multi-scale candidates.

    Replaces MixedQuerySelector:
      - Old: topk(scores) + detach → scorer gradient blocked
      - New: learnable slot queries attend to all K candidates via cross-attn
             → fully differentiable, no topk, no detach

    Output : [B*T, n_slots, d_model]
    """

    def __init__(self, d_model: int = 256, n_slots: int = 3,
                 n_candidates: int = 7, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.n_slots = n_slots

        # Learnable slot content queries with spatial prior
        # For n_slots=3 → 0°, 120°, 240° initial bias
        slot_init = torch.randn(n_slots, d_model) * 0.02
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

        # Cross-attention: slot queries (Q) attend to candidates (K, V)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, candidates: list) -> torch.Tensor:
        """
        Parameters
        ----------
        candidates : list of n_candidates tensors, each [B, T, d]

        Returns
        -------
        [B*T, n_slots, d]
        """
        cand_stack = torch.stack(candidates, dim=2)   # [B, T, K, d]
        B, T, K, d = cand_stack.shape
        kv = cand_stack.reshape(B * T, K, d)          # [B*T, K, d]

        q = self.slot_queries.unsqueeze(0).expand(B * T, -1, -1)  # [B*T, S, d]

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q = self.norm1(q + attn_out)

        # Add slot-specific spatial embedding to encourage diversity
        spatial_emb = self.slot_spatial_embed.unsqueeze(0).expand(B * T, -1, -1)
        q = q + spatial_emb

        q = self.norm2(q + self.ffn(q))
        return q   # [B*T, n_slots, d]


# =============================================================================
# ContrastiveDeNoising
# =============================================================================

class ContrastiveDeNoising(nn.Module):
    """Generates DN queries from GT annotations (training only).

    Fix vs original: each frame gets its own random class permutation for
    negative queries (was: all B*T frames shared one permutation).
    """

    def __init__(self, d_model: int = 256, n_classes: int = 300,
                 n_dn_groups: int = 3,
                 noise_scale_pos: float = 0.2,
                 noise_scale_neg: float = 0.8):
        super().__init__()
        self.d_model         = d_model
        self.n_classes       = n_classes
        self.n_dn_groups     = n_dn_groups
        self.noise_scale_pos = noise_scale_pos
        self.noise_scale_neg = noise_scale_neg

        self.class_embed = nn.Embedding(n_classes + 1, d_model,
                                        padding_idx=n_classes)
        self.doa_embed   = nn.Linear(3, d_model, bias=False)
        self.query_proj  = nn.Linear(d_model, d_model)

    def _perturb_doa(self, doa: torch.Tensor, scale: float) -> torch.Tensor:
        noise = torch.randn_like(doa) * scale
        return F.normalize(doa + noise, p=2, dim=-1)

    def forward(self, gt_cls, gt_doa, gt_loud, gt_mask):
        """
        Parameters
        ----------
        gt_cls  : [B, T, S]      int64
        gt_doa  : [B, T, S, 3]   float32
        gt_loud : [B, T, S]      (unused, kept for API consistency)
        gt_mask : [B, T, S]      bool

        Returns
        -------
        dn_queries  : [B*T, G*2*S, d]
        pos_targets : {cls [B*T, G, S], doa [B*T, G, S, 3]}
        neg_targets : {cls [B*T, G, S], doa [B*T, G, S, 3]}
        S_dn        : int   G * 2 * S
        """
        B, T, S = gt_cls.shape
        G  = self.n_dn_groups
        BT = B * T

        gt_cls_flat  = gt_cls.reshape(BT, S)
        gt_doa_flat  = gt_doa.reshape(BT, S, 3)
        gt_mask_flat = gt_mask.reshape(BT, S)

        cls_safe = gt_cls_flat.clone()
        cls_safe[cls_safe < 0] = self.n_classes

        S_dn = G * 2 * S

        dn_q_list  = []
        pos_cls_l, pos_doa_l = [], []
        neg_cls_l, neg_doa_l = [], []

        for _ in range(G):
            # Positive DN queries (small noise)
            doa_pos = self._perturb_doa(gt_doa_flat, self.noise_scale_pos)
            cls_emb = self.class_embed(cls_safe)
            doa_emb = self.doa_embed(doa_pos)
            q_pos   = self.query_proj(cls_emb + doa_emb)
            q_pos   = q_pos * gt_mask_flat.unsqueeze(-1).float()
            dn_q_list.append(q_pos)
            pos_cls_l.append(cls_safe.clone())
            pos_doa_l.append(doa_pos.clone())

            # Negative DN queries (large noise, per-frame class shuffle)
            doa_neg = self._perturb_doa(gt_doa_flat, self.noise_scale_neg)
            # Per-frame independent permutation (fix: was one shared perm)
            noise    = torch.rand(BT, S, device=gt_cls.device)
            perm     = noise.argsort(dim=-1)            # [BT, S]
            cls_neg  = cls_safe.gather(1, perm)         # [BT, S]
            cls_emb_n = self.class_embed(cls_neg)
            doa_emb_n = self.doa_embed(doa_neg)
            q_neg     = self.query_proj(cls_emb_n + doa_emb_n)
            q_neg     = q_neg * gt_mask_flat.unsqueeze(-1).float()
            dn_q_list.append(q_neg)
            neg_cls_l.append(cls_neg.clone())
            neg_doa_l.append(doa_neg.clone())

        dn_queries = torch.cat(dn_q_list, dim=1)   # [BT, G*2*S, d]

        pos_targets = {
            'cls': torch.stack(pos_cls_l, dim=1),   # [BT, G, S]
            'doa': torch.stack(pos_doa_l, dim=1),
        }
        neg_targets = {
            'cls': torch.stack(neg_cls_l, dim=1),
            'doa': torch.stack(neg_doa_l, dim=1),
        }

        return dn_queries, pos_targets, neg_targets, S_dn


# =============================================================================
# SpatialBeamformingMemory
# =============================================================================

class SpatialBeamformingMemory(nn.Module):
    """Fixed angular grid → spatial tokens that fuse with encoder output.

    Builds 72 unit vectors on a sphere (36 az × 2 el) as a fixed directional
    grid, projects them to d_model, and cross-attends each frame's encoder
    representation.  The resulting tokens carry both directional identity and
    scene-specific acoustic information, giving the decoder an explicit spatial
    compass for multi-source localisation.

    Parameters
    ----------
    d_model : int   feature dimension (256)
    n_az    : int   azimuth bins (36 → 10° resolution)
    n_el    : int   elevation bins (2 → ±30°)
    n_heads : int   attention heads
    """

    def __init__(self, d_model: int = 256, n_az: int = 36, n_el: int = 2,
                 n_heads: int = 8):
        super().__init__()
        self.n_dirs = n_az * n_el

        # ── Build fixed angular grid ──────────────────────────────────────────
        # Azimuths: 0, 10, 20, …, 350 degrees (uniform ring)
        # Elevations: −30°, +30°
        azimuths   = [2 * math.pi * i / n_az for i in range(n_az)]
        elevations = [-30.0 * math.pi / 180.0, 30.0 * math.pi / 180.0]

        dirs = []
        for el in elevations:
            for az in azimuths:
                x = math.cos(el) * math.cos(az)
                y = math.cos(el) * math.sin(az)
                z = math.sin(el)
                dirs.append([x, y, z])

        direction_grid = torch.tensor(dirs, dtype=torch.float32)  # [n_dirs, 3]
        self.register_buffer('direction_grid', direction_grid)

        # ── Spatial projection: direction → d_model ───────────────────────────
        self.spatial_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Fusion: spatial tokens cross-attend encoder output ────────────────
        self.fusion_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_out : [B, T, d]   encoder summary per frame

        Returns
        -------
        [B*T, n_dirs, d]   direction-aware spatial memory tokens
        """
        B, T, d = enc_out.shape
        BT = B * T

        # Project direction grid → [n_dirs, d] → [B*T, n_dirs, d]
        spatial_q = self.spatial_proj(self.direction_grid)            # [n_dirs, d]
        spatial_q = spatial_q.unsqueeze(0).expand(BT, -1, -1)        # [B*T, n_dirs, d]

        # KV: each frame's encoder output  [B*T, 1, d]
        enc_flat = enc_out.reshape(BT, 1, d)

        # Cross-attend: spatial directions query encoder summary per frame
        attn_out, _ = self.fusion_attn(spatial_q, enc_flat, enc_flat,
                                       need_weights=False)
        spatial_tokens = self.norm(spatial_q + attn_out)              # [B*T, n_dirs, d]

        return spatial_tokens


# =============================================================================
# SlotCompetitionLayer  (replaces DecoderLayer)
# =============================================================================

class SlotCompetitionLayer(nn.Module):
    """Single decoder layer with dual-memory cross-attention.

    Processing order (post-norm):
      1. self-attn          — inter-slot competition; slots influence each other
      2. spatial cross-attn — Q=slots, KV=spatial_memory (72 direction tokens)
      3. freq cross-attn    — Q=slots, KV=freq_memory   (11 scale tokens)
      4. FFN

    The explicit inter-slot self-attention forces diversity: once one slot
    attends heavily to a spatial direction the other slots see that and are
    discouraged from attending the same way.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()

        # 1. inter-slot self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # 2. spatial cross-attention
        self.spatial_cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        # 3. frequency cross-attention
        self.freq_cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                     dropout=dropout,
                                                     batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

        # 4. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, slots: torch.Tensor,
                freq_memory: torch.Tensor,
                spatial_memory: torch.Tensor,
                self_attn_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        slots          : [B*T, S_total, d]
        freq_memory    : [B*T, 11, d]
        spatial_memory : [B*T, n_dirs, d]
        self_attn_mask : [S_total, S_total] or None

        Returns
        -------
        slots            : [B*T, S_total, d]
        spatial_weights  : [B*T, S_total, n_dirs]  (for optional diagnostics)
        """
        # 1. inter-slot self-attention
        sa_out, _ = self.self_attn(slots, slots, slots,
                                   attn_mask=self_attn_mask,
                                   need_weights=False)
        slots = self.norm1(slots + self.drop1(sa_out))

        # 2. spatial cross-attention
        spa_out, spa_weights = self.spatial_cross_attn(
            slots, spatial_memory, spatial_memory, need_weights=True
        )
        slots = self.norm2(slots + self.drop2(spa_out))

        # 3. frequency cross-attention
        freq_out, _ = self.freq_cross_attn(
            slots, freq_memory, freq_memory, need_weights=False
        )
        slots = self.norm3(slots + self.drop3(freq_out))

        # 4. FFN
        slots = self.norm4(slots + self.ffn(slots))

        return slots, spa_weights   # spa_weights: [B*T, S_total, n_dirs]


# =============================================================================
# IterativeRefinementDecoder
# =============================================================================

class IterativeRefinementDecoder(nn.Module):
    """Decoder with iterative DOA-based positional refinement.

    Each layer is a SlotCompetitionLayer that attends to both a spatial memory
    (72 direction-aware tokens) and a frequency memory (7 multi-scale tokens).
    After each intermediate layer a lightweight head predicts a rough DOA vector
    from matching queries and injects it as a positional embedding into the next
    layer for iterative geometric correction.

    DN queries (slots S..S_total) are untouched by refinement.

    Returns
    -------
    list of n_layers tensors [B*T, S_total, d]
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, n_slots: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_slots  = n_slots
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            SlotCompetitionLayer(d_model, n_heads,
                                 ffn_dim=d_model * 4,
                                 dropout=dropout)
            for _ in range(n_layers)
        ])

        # DOA prediction heads for positional feedback (n_layers-1 refinements)
        self.doa_refine = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 3),
            )
            for _ in range(n_layers - 1)
        ])

        # DOA unit vector → positional embedding
        self.doa_pos_enc = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, queries: torch.Tensor,
                freq_memory: torch.Tensor,
                spatial_memory: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> list:
        """
        Parameters
        ----------
        queries        : [B*T, S_total, d]   initial queries (matching + optional DN)
        freq_memory    : [B*T, 11, d]        encoder multi-scale memory
        spatial_memory : [B*T, n_dirs, d]    direction-aware spatial memory
        attn_mask      : [S_total, S_total]  or None

        Returns
        -------
        list of n_layers tensors [B*T, S_total, d]
        """
        outputs = []
        x = queries
        S = self.n_slots

        for i, layer in enumerate(self.layers):
            x, _ = layer(x, freq_memory, spatial_memory,
                         self_attn_mask=attn_mask)
            outputs.append(x)

            if i < self.n_layers - 1:
                # Predict DOA from current matching queries
                doa_raw  = self.doa_refine[i](x[:, :S, :])   # [B*T, S, 3]
                doa_unit = F.normalize(doa_raw, dim=-1)
                pos_emb  = self.doa_pos_enc(doa_unit)         # [B*T, S, d]

                # Inject into matching queries; leave DN queries unchanged
                x = torch.cat([
                    x[:, :S, :] + pos_emb,
                    x[:, S:, :],
                ], dim=1)

        return outputs
