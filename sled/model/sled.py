#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Full Model
=====================
SLEDv3 end-to-end: waveform (or pre-computed features) → per-frame detections.

Architecture
------------
  AudioPreprocessor → [B, 6, 64, T]
  SLEDEncoder       → (multi_scale_feats, enc_out)
  MixedQuerySelector → queries [B*T, 5, d]
  ContrastiveDeNoising (training) → dn_queries
  LookForwardTwiceDecoder → list of [B*T, S_total, d]
  DetectionHeads (per layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessor import AudioPreprocessor
from .encoder      import SLEDEncoder
from .decoder      import (
    MixedQuerySelector,
    ContrastiveDeNoising,
    LookForwardTwiceDecoder,
)
from .heads        import DetectionHeads


class SLEDv3(nn.Module):
    """SLED v3 with DINO-style improvements (DN + LFT decoder).

    Parameters
    ----------
    sofa_path           : path to SOFA HRTF file
    d_model             : feature dimension (256)
    n_slots             : maximum simultaneous sources per frame (3)
    n_classes           : real sound classes only, no empty class (e.g. 209)
    n_decoder_layers    : number of Look-Forward-Twice decoder layers (4)
    n_conformer_layers  : number of CausalConformerBlocks in encoder (4)
    precompute_features : if True, skip preprocessor in forward (input is [B,6,64,T])
    """

    N_CANDIDATES = 7   # 6 sub-band + 1 enc_out

    def __init__(self, sofa_path: str, d_model: int = 256,
                 n_slots: int = 3, n_classes: int = 301,
                 n_decoder_layers: int = 4, n_conformer_layers: int = 4,
                 precompute_features: bool = False):
        super().__init__()

        self.n_slots             = n_slots
        self.n_classes           = n_classes
        self.precompute_features = precompute_features

        self.preprocessor = AudioPreprocessor(sofa_path)
        self.encoder      = SLEDEncoder(
            in_channels  = 5,
            d_model      = d_model,
            n_conformer  = n_conformer_layers,
        )
        self.query_selector = MixedQuerySelector(
            d_model      = d_model,
            n_slots      = n_slots,
            n_candidates = self.N_CANDIDATES,
        )
        # n_classes = real sound classes only (no empty); DN uses the same space
        self.denoising = ContrastiveDeNoising(
            d_model     = d_model,
            n_classes   = n_classes,
            n_dn_groups = 3,
        )
        self.decoder = LookForwardTwiceDecoder(
            d_model  = d_model,
            n_layers = n_decoder_layers,
            n_slots  = n_slots,
        )
        self.heads = DetectionHeads(
            d_model   = d_model,
            n_classes = n_classes,
            n_slots   = n_slots,
        )

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_dn_mask(S: int, S_dn: int,
                       device: torch.device) -> torch.Tensor:
        """Build combined self-attention mask for matching + DN queries.

        Convention: True = blocked.
        Matching queries cannot see DN queries and vice versa.
        """
        total = S + S_dn
        mask  = torch.zeros(total, total, dtype=torch.bool, device=device)
        # matching → DN: blocked
        mask[:S, S:] = True
        # DN → matching: blocked
        mask[S:, :S] = True
        return mask

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, audio_or_feat: torch.Tensor,
                gt: dict | None = None) -> dict:
        """
        Parameters
        ----------
        audio_or_feat : [B, 2, N] stereo waveform  OR  [B, 5, 64, T] features
        gt            : optional dict (training only) with keys:
                          'cls'  [B, T, 5] int64
                          'doa'  [B, T, 5, 3] float32
                          'loud' [B, T, 5] float32
                          'mask' [B, T, 5] bool

        Returns
        -------
        result dict:
          'layer_preds'    : list of n_layers dicts, each with
                             class_logits, doa_vec, loudness, confidence
          'dn_layer_preds' : (training) list of n_layers dicts
          'pos_targets'    : (training) dict {cls, doa}
          'neg_targets'    : (training) dict {cls, doa}
          'S_dn'           : (training) int
        """
        # ── Feature extraction ────────────────────────────────────────────────
        if audio_or_feat.dim() == 3 and not self.precompute_features:
            feat, hrtf_ch = self.preprocessor(audio_or_feat)
            # feat: [B, 5, 64, T]   hrtf_ch: [B, 64, T_stft]
        else:
            feat    = audio_or_feat   # [B, 5, 64, T]  (pre-computed)
            hrtf_ch = None

        B, C, F_mel, T = feat.shape

        # ── Encoder ──────────────────────────────────────────────────────────
        multi_scale, enc_out = self.encoder(feat, hrtf_ch)
        # multi_scale : list of 7 × [B, T, d]
        # enc_out     : [B, T, d]

        # ── Query selection ───────────────────────────────────────────────────
        queries = self.query_selector(multi_scale)   # [B*T, S, d]

        # ── Align T with GT if provided (STFT may produce T+1 frames) ─────────
        if gt is not None:
            T_gt = gt['cls'].shape[1]
            if T != T_gt:
                T = T_gt
                enc_out = enc_out[:, :T, :]
                multi_scale = [f[:, :T, :] for f in multi_scale]
                queries = queries.reshape(B, -1, self.n_slots, queries.shape[-1])
                queries = queries[:, :T].reshape(B * T, self.n_slots, -1)

        # ── Memory (encoder output as cross-attention key-value) ──────────────
        # Reshape enc_out → [B*T, 1, d]
        memory = enc_out.reshape(B * T, 1, -1)

        # ── Denoising (training only) ─────────────────────────────────────────
        S_dn     = 0
        attn_mask = None

        if gt is not None and self.training:
            dn_queries, pos_targets, neg_targets, S_dn = self.denoising(
                gt['cls'], gt['doa'], gt['loud'], gt['mask']
            )
            all_queries = torch.cat([queries, dn_queries], dim=1)
            attn_mask   = self._build_dn_mask(
                S=self.n_slots, S_dn=S_dn, device=feat.device
            )
        else:
            all_queries = queries

        # ── Decoder ───────────────────────────────────────────────────────────
        all_layer_outputs = self.decoder(all_queries, memory, attn_mask)
        # List of n_layers tensors [B*T, S_total, d]

        # ── Detection heads for matching queries ─────────────────────────────
        layer_preds = []
        for layer_out in all_layer_outputs:
            match_out = layer_out[:, :self.n_slots, :]   # [B*T, S, d]
            match_out = match_out.reshape(B, T, self.n_slots, -1)
            pred      = self.heads(match_out, B, T)
            layer_preds.append(pred)

        result = {'layer_preds': layer_preds}

        # ── DN heads ─────────────────────────────────────────────────────────
        if gt is not None and self.training and S_dn > 0:
            dn_layer_preds = []
            for layer_out in all_layer_outputs:
                dn_out  = layer_out[:, self.n_slots:, :]   # [B*T, S_dn, d]
                dn_out  = dn_out.reshape(B, T, S_dn, -1)
                dn_pred = self.heads(dn_out, B, T)
                dn_layer_preds.append(dn_pred)

            result['dn_layer_preds'] = dn_layer_preds
            result['pos_targets']    = pos_targets
            result['neg_targets']    = neg_targets
            result['S_dn']           = S_dn

        return result
