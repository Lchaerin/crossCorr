#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Auxiliary Losses
===========================
  SlotDiversityLoss  : repulsion between active slot DOA predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SlotDiversityLoss(nn.Module):
    """Repulsion between active slot DOA predictions.

    For each pair of slots (i, j) that are both active in a frame, penalises
    high cosine similarity between their predicted DOA unit vectors.  This
    encourages slots to attend to different directions and prevents collapse
    to a single dominant direction when multiple sources are present.
    """

    def forward(self, doa_vecs: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        doa_vecs : [B, T, S, 3]   predicted DOA unit vectors
        mask     : [B, T, S]      bool — True for GT-active (matched) slots

        Returns
        -------
        scalar diversity loss ≥ 0
        """
        S = doa_vecs.shape[2]
        loss = torch.tensor(0.0, device=doa_vecs.device, requires_grad=True)
        count = 0

        for i in range(S):
            for j in range(i + 1, S):
                both = mask[:, :, i] & mask[:, :, j]   # [B, T]
                if both.any():
                    cos = F.cosine_similarity(
                        doa_vecs[:, :, i], doa_vecs[:, :, j], dim=-1
                    )   # [B, T]
                    loss = loss + cos[both].clamp(min=0).mean()
                    count += 1

        return loss / max(count, 1)
