#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Decoder
==================
Components:
  MixedQuerySelector    — selects top-5 queries from 7 candidates
  ContrastiveDeNoising  — generates DN queries from GT labels (training only)
  DecoderLayer          — self-attn + cross-attn + FFN (post-norm)
  LookForwardTwiceDecoder — 4 decoder layers with Look Forward Twice
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MixedQuerySelector
# =============================================================================

class MixedQuerySelector(nn.Module):
    """Selects the top-n_slots most informative candidate features as queries.

    7 candidates enter (6 sub-band + enc_out).
    Each candidate is scored with a lightweight linear head.
    The top-n_slots are chosen, their features become anchor positional
    embeddings, and learnable content queries are added.

    Output : [B*T, n_slots, d_model]
    """

    def __init__(self, d_model: int = 256, n_slots: int = 3,
                 n_candidates: int = 7):
        super().__init__()
        self.d_model      = d_model
        self.n_slots      = n_slots
        self.n_candidates = n_candidates

        # Learnable content queries [n_slots, d_model]
        self.content_queries = nn.Parameter(
            torch.randn(n_slots, d_model) * 0.02
        )

        # Scorer: maps each candidate feature → scalar score
        self.scorer = nn.Linear(d_model, 1, bias=True)

        # Project anchor positional embedding to d_model (already d_model, kept for completeness)
        self.anchor_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, candidates: list) -> torch.Tensor:
        """
        Parameters
        ----------
        candidates : list of n_candidates tensors, each [B, T, d_model]

        Returns
        -------
        queries : [B*T, n_slots, d_model]
        """
        # Stack candidates → [B, T, n_candidates, d_model]
        cand_stack = torch.stack(candidates, dim=2)  # [B, T, K, d]
        B, T, K, d = cand_stack.shape

        # Score each candidate
        scores = self.scorer(cand_stack).squeeze(-1)  # [B, T, K]

        # Select top-n_slots indices
        _, top_idx = scores.topk(self.n_slots, dim=-1, sorted=True)  # [B,T,S]

        # Gather anchor positional embeddings (detached — no gradient through selection)
        top_idx_exp = top_idx.unsqueeze(-1).expand(B, T, self.n_slots, d)
        anchor_pos  = cand_stack.gather(2, top_idx_exp)  # [B,T,S,d]
        anchor_pos  = anchor_pos.detach()

        # Project anchor
        anchor_pos  = self.anchor_proj(anchor_pos)  # [B,T,S,d]

        # Expand content queries → [B, T, S, d]
        content = self.content_queries.view(1, 1, self.n_slots, d).expand(B, T, -1, -1)

        # Mixed query = content + anchor positional embedding
        queries = content + anchor_pos  # [B, T, S, d]

        # Reshape to [B*T, S, d]
        queries = queries.reshape(B * T, self.n_slots, d)
        return queries


# =============================================================================
# ContrastiveDeNoising
# =============================================================================

class ContrastiveDeNoising(nn.Module):
    """Generates denoising (DN) queries from GT annotations.

    Produces n_dn_groups × n_slots positive and negative queries by adding
    noise to GT DOA vectors and class labels.

    Parameters
    ----------
    d_model         : feature dimension
    n_classes       : number of real sound classes (not counting "empty")
    n_dn_groups     : number of DN groups (set externally after each phase change)
    noise_scale_pos : perturbation magnitude for positive (small noise)
    noise_scale_neg : perturbation magnitude for negative (large noise)
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

        # Embed class label → d_model
        self.class_embed = nn.Embedding(n_classes + 1, d_model,
                                        padding_idx=n_classes)
        # Embed DOA unit vector → d_model
        self.doa_embed   = nn.Linear(3, d_model, bias=False)

        # Combine class + DOA embedding → query
        self.query_proj  = nn.Linear(d_model, d_model)

    def _perturb_doa(self, doa: torch.Tensor, scale: float) -> torch.Tensor:
        """Add Gaussian noise to DOA vector and renormalise."""
        noise   = torch.randn_like(doa) * scale
        perturb = doa + noise
        # Renormalise; avoid zero-vector
        return F.normalize(perturb, p=2, dim=-1)

    def forward(self, gt_cls, gt_doa, gt_loud, gt_mask):
        """
        Parameters
        ----------
        gt_cls  : [B, T, S]      int64  (class ids, -1 = inactive)
        gt_doa  : [B, T, S, 3]   float32
        gt_loud : [B, T, S]      float32  (unused here, for signature consistency)
        gt_mask : [B, T, S]      bool

        Returns
        -------
        dn_queries    : [B*T, S_dn, d_model]
        pos_targets   : dict {cls, doa}  each [B*T, n_dn_groups, S, …]
        neg_targets   : dict {cls, doa}  each [B*T, n_dn_groups, S, …]
        S_dn          : int   n_dn_groups * 2 * S   (pos + neg per group)
        """
        B, T, S = gt_cls.shape
        d       = self.d_model
        G       = self.n_dn_groups
        BT      = B * T

        # Flatten batch & time
        gt_cls_flat  = gt_cls.reshape(BT, S)              # [BT, S]
        gt_doa_flat  = gt_doa.reshape(BT, S, 3)           # [BT, S, 3]
        gt_mask_flat = gt_mask.reshape(BT, S)              # [BT, S] bool

        # Replace inactive class ids (-1) with padding_idx (n_classes)
        cls_safe = gt_cls_flat.clone()
        cls_safe[cls_safe < 0] = self.n_classes

        S_dn = G * 2 * S  # total DN slots

        # Pre-allocate outputs
        dn_q_list  = []
        pos_cls_l, pos_doa_l = [], []
        neg_cls_l, neg_doa_l = [], []

        for _ in range(G):
            # ── Positive DN queries (small noise) ────────────────────────────
            doa_pos = self._perturb_doa(gt_doa_flat, self.noise_scale_pos)
            cls_emb = self.class_embed(cls_safe)          # [BT, S, d]
            doa_emb = self.doa_embed(doa_pos)             # [BT, S, d]
            q_pos   = self.query_proj(cls_emb + doa_emb)  # [BT, S, d]
            # Zero-out inactive slots
            q_pos   = q_pos * gt_mask_flat.unsqueeze(-1).float()
            dn_q_list.append(q_pos)

            pos_cls_l.append(cls_safe.clone())
            pos_doa_l.append(doa_pos.clone())

            # ── Negative DN queries (large noise) ────────────────────────────
            doa_neg = self._perturb_doa(gt_doa_flat, self.noise_scale_neg)
            # Randomly shuffle class labels for negatives
            perm     = torch.randperm(S, device=gt_cls.device)
            cls_neg  = cls_safe[:, perm]
            cls_emb_n = self.class_embed(cls_neg)
            doa_emb_n = self.doa_embed(doa_neg)
            q_neg     = self.query_proj(cls_emb_n + doa_emb_n)
            q_neg     = q_neg * gt_mask_flat.unsqueeze(-1).float()
            dn_q_list.append(q_neg)

            neg_cls_l.append(cls_neg.clone())
            neg_doa_l.append(doa_neg.clone())

        # Concatenate groups: [BT, G*2*S, d]
        dn_queries = torch.cat(dn_q_list, dim=1)

        pos_targets = {
            'cls': torch.stack(pos_cls_l, dim=1),   # [BT, G, S]
            'doa': torch.stack(pos_doa_l, dim=1),   # [BT, G, S, 3]
        }
        neg_targets = {
            'cls': torch.stack(neg_cls_l, dim=1),
            'doa': torch.stack(neg_doa_l, dim=1),
        }

        return dn_queries, pos_targets, neg_targets, S_dn


# =============================================================================
# DecoderLayer
# =============================================================================

class DecoderLayer(nn.Module):
    """Single decoder layer: self-attention → cross-attention → FFN.

    Uses post-norm (LayerNorm after residual connection).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        # Self-attention
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.norm1      = nn.LayerNorm(d_model)
        self.drop1      = nn.Dropout(dropout)

        # Cross-attention (queries attend to memory = enc_out)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        self.drop2      = nn.Dropout(dropout)

        # Feed-forward
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                self_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        tgt              : [BT, S, d]   query tokens
        memory           : [BT, M, d]   encoder memory
        self_attn_mask   : [S, S] or None   boolean mask (True = blocked)

        Returns
        -------
        [BT, S, d]
        """
        # Self-attention (post-norm)
        sa_out, _ = self.self_attn(tgt, tgt, tgt,
                                   attn_mask=self_attn_mask,
                                   need_weights=False)
        tgt = self.norm1(tgt + self.drop1(sa_out))

        # Cross-attention
        ca_out, _ = self.cross_attn(tgt, memory, memory, need_weights=False)
        tgt = self.norm2(tgt + self.drop2(ca_out))

        # FFN
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt


# =============================================================================
# LookForwardTwiceDecoder
# =============================================================================

class LookForwardTwiceDecoder(nn.Module):
    """4-layer decoder with Look Forward Twice (LFT) anchor refinement.

    Each layer i outputs a prediction; layer i+1 uses the refined anchor
    positions from layer i (anchor_delta is applied without detach, allowing
    gradient flow through the anchor refinement).

    After each layer, a lightweight anchor-delta head refines the query
    embeddings by adding a residual update.

    Returns
    -------
    list of [B*T, S_total, d] — one tensor per layer
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, n_slots: int = 3):
        super().__init__()
        self.n_layers = n_layers
        self.n_slots  = n_slots

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, ffn_dim=d_model * 2)
            for _ in range(n_layers)
        ])

        # Per-layer anchor-delta head: predicts a residual update for the query
        # This implements the "Look Forward Twice" refinement
        self.anchor_delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_layers)
        ])

        # Layer norm applied after LFT refinement
        self.lft_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def forward(self, queries: torch.Tensor, memory: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> list:
        """
        Parameters
        ----------
        queries   : [BT, S_total, d]   initial queries (matching + DN)
        memory    : [BT, M, d]         encoder memory
        attn_mask : [S_total, S_total] or None

        Returns
        -------
        list of n_layers tensors, each [BT, S_total, d]
        """
        outputs   = []
        x         = queries

        for i, (layer, delta_head, lft_norm) in enumerate(
            zip(self.layers, self.anchor_delta_heads, self.lft_norms)
        ):
            x = layer(x, memory, self_attn_mask=attn_mask)

            # Look Forward Twice: compute anchor delta and add as residual
            # (no detach on the anchor — gradient flows through refinement)
            delta = delta_head(x)   # [BT, S, d]
            x     = lft_norm(x + delta)

            outputs.append(x)

        return outputs
