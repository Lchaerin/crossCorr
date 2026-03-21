#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Detection Heads
==========================
Maps per-slot features → class logits, DOA unit-vector, loudness, confidence.

Input  : [B, T, S, d_model]  (or any shape where last dim = d_model)
Output : dict
    class_logits : [B, T, S, n_classes]   (n_classes = n_real + 1 empty class)
    doa_vec      : [B, T, S, 3]           unit vector
    loudness     : [B, T, S]              scalar (dB)
    confidence   : [B, T, S]             scalar (logit for "this slot is active")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHeads(nn.Module):
    """Per-slot detection heads.

    Parameters
    ----------
    d_model   : input feature dimension
    n_classes : total number of output classes including "empty" (e.g. 301)
    n_slots   : maximum simultaneous sources per frame
    """

    def __init__(self, d_model: int = 256, n_classes: int = 301,
                 n_slots: int = 3):
        super().__init__()
        self.d_model   = d_model
        self.n_classes = n_classes
        self.n_slots   = n_slots

        # Classification head: one logit per class
        self.class_head = nn.Linear(d_model, n_classes)

        # DOA head: 256 → 256 → 3, then L2-normalised
        self.doa_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

        # Loudness head: scalar dB estimate
        self.loud_head = nn.Linear(d_model, 1)

        # Confidence head: scalar (probability of slot being active)
        self.conf_head = nn.Linear(d_model, 1)

    def forward(self, slot_features: torch.Tensor,
                B: int, T: int) -> dict:
        """
        Parameters
        ----------
        slot_features : [B, T, S, d_model]  — already reshaped by the caller
        B, T          : batch size and time steps (passed for clarity / assertion)

        Returns
        -------
        dict with keys: class_logits, doa_vec, loudness, confidence
        """
        # x may arrive as [B, T, S, d] or [B*T, S, d] (caller reshapes before)
        x = slot_features   # [B, T, S, d]

        # Class logits
        class_logits = self.class_head(x)               # [B, T, S, n_classes]

        # DOA unit vector
        doa_raw = self.doa_head(x)                       # [B, T, S, 3]
        doa_vec = F.normalize(doa_raw, p=2, dim=-1)

        # Loudness (scalar)
        loudness = self.loud_head(x).squeeze(-1)         # [B, T, S]

        # Confidence (logit; apply sigmoid at loss computation)
        confidence = self.conf_head(x).squeeze(-1)       # [B, T, S]

        return {
            'class_logits': class_logits,
            'doa_vec'     : doa_vec,
            'loudness'    : loudness,
            'confidence'  : confidence,
        }
