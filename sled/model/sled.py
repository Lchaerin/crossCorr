#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Full Model
=====================
SLEDv3 end-to-end: waveform (or pre-computed features) → per-frame detections.

Architecture
------------
  AudioPreprocessor        → feat [B, 5, 64, T],  hrtf_ch [B, 64, 32]
  SLEDEncoder              → (multi_scale_feats 7×[B,T,d],  enc_out [B,T,d])
  CrossAttentionQuerySelector → queries [B*T, S, d]
  memory = stack(multi_scale) → [B*T, 7, d]   (7 meaningful tokens, not 1)
  ContrastiveDeNoising (training) → dn_queries
  IterativeRefinementDecoder → list of [B*T, S_total, d]
  DetectionHeads (per layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessor import AudioPreprocessor
from .encoder      import SLEDEncoder
from .decoder      import (
    CrossAttentionQuerySelector,
    ContrastiveDeNoising,
    IterativeRefinementDecoder,
)
from .heads        import DetectionHeads


class SLEDv3(nn.Module):
    """SLED v3 — cross-attention queries, 7-token memory, iterative DOA refinement.

    Parameters
    ----------
    sofa_path           : path to SOFA HRTF file
    d_model             : feature dimension (256)
    n_slots             : maximum simultaneous sources per frame (3)
    n_classes           : real sound classes only, no empty class (e.g. 209)
    n_decoder_layers    : number of IterativeRefinement decoder layers (4)
    n_conformer_layers  : number of CausalConformerBlocks in encoder (4)
    precompute_features : if True, skip preprocessor (input must be [B,5,64,T])
    use_hrtf_corr       : if False, skip HRTF cross-corr heatmap (ablation)
    use_ild             : if False, drop ILD channel from input (ablation)
    use_ipd             : if False, drop IPD channels (sin/cos) from input (ablation)
    """

    N_CANDIDATES = 7   # 6 sub-band + 1 enc_out

    def __init__(self, sofa_path: str, d_model: int = 256,
                 n_slots: int = 3, n_classes: int = 209,
                 n_decoder_layers: int = 4, n_conformer_layers: int = 4,
                 precompute_features: bool = False,
                 use_hrtf_corr: bool = True,
                 use_ild: bool = True,
                 use_ipd: bool = True):
        super().__init__()

        self.n_slots             = n_slots
        self.n_classes           = n_classes
        self.precompute_features = precompute_features
        self.use_hrtf_corr       = use_hrtf_corr
        self.use_ild             = use_ild
        self.use_ipd             = use_ipd

        self.preprocessor = AudioPreprocessor(sofa_path,
                                              use_hrtf_corr=use_hrtf_corr,
                                              use_ild=use_ild,
                                              use_ipd=use_ipd)
        # in_channels: L-mel + R-mel always, plus optional ILD and IPD (×2)
        in_channels = 2 + int(use_ild) + 2 * int(use_ipd)
        self.encoder      = SLEDEncoder(
            in_channels   = in_channels,
            d_model       = d_model,
            n_conformer   = n_conformer_layers,
            use_hrtf_corr = use_hrtf_corr,
        )
        self.query_selector = CrossAttentionQuerySelector(
            d_model      = d_model,
            n_slots      = n_slots,
            n_candidates = self.N_CANDIDATES,
        )
        self.denoising = ContrastiveDeNoising(
            d_model     = d_model,
            n_classes   = n_classes,
            n_dn_groups = 3,
        )
        self.decoder = IterativeRefinementDecoder(
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
        """Self-attention mask: matching ↔ DN queries are mutually blocked."""
        total = S + S_dn
        mask  = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:S, S:] = True   # matching → DN: blocked
        mask[S:, :S] = True   # DN → matching: blocked
        return mask

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, audio_or_feat: torch.Tensor,
                gt: dict | None = None) -> dict:
        """
        Parameters
        ----------
        audio_or_feat : [B, 2, N] stereo waveform  OR  [B, 5, 64, T] features
        gt            : optional dict (training only):
                          'cls'  [B, T, S] int64
                          'doa'  [B, T, S, 3] float32
                          'loud' [B, T, S] float32
                          'mask' [B, T, S] bool

        Returns
        -------
        dict:
          'layer_preds'    : list of n_layers dicts {class_logits, doa_vec,
                             loudness, confidence}
          'dn_layer_preds' : (training) list of n_layers dicts
          'pos_targets'    : (training) {cls, doa}
          'neg_targets'    : (training) {cls, doa}
          'S_dn'           : (training) int
        """
        # ── Feature extraction ────────────────────────────────────────────────
        if audio_or_feat.dim() == 3 and not self.precompute_features:
            feat, hrtf_ch = self.preprocessor(audio_or_feat)
            # feat: [B, 5, 64, T]   hrtf_ch: [B, 64, 32]
        else:
            feat    = audio_or_feat
            hrtf_ch = None

        B, C, F_mel, T = feat.shape

        # ── Encoder ──────────────────────────────────────────────────────────
        multi_scale, enc_out = self.encoder(feat, hrtf_ch)
        # multi_scale : 7 × [B, T, d]
        # enc_out     : [B, T, d]

        # ── T alignment (STFT may produce T+1 frames; must match GT length) ──
        # Done BEFORE query selection so shapes are consistent throughout.
        if gt is not None:
            T_gt = gt['cls'].shape[1]
            if T != T_gt:
                T           = T_gt
                enc_out     = enc_out[:, :T, :]
                multi_scale = [f[:, :T, :] for f in multi_scale]

        d = enc_out.shape[-1]

        # ── Query selection ───────────────────────────────────────────────────
        # Fully differentiable: learnable slots attend to 7 candidates.
        queries = self.query_selector(multi_scale)   # [B*T, S, d]

        # ── Memory: all 7 multi-scale tokens (not just 1) ─────────────────────
        # Gives the decoder rich multi-frequency context per frame.
        memory = torch.stack(multi_scale, dim=2).reshape(B * T, 7, d)  # [B*T, 7, d]

        # ── Denoising (training only) ─────────────────────────────────────────
        S_dn      = 0
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
        # List of n_layers × [B*T, S_total, d]

        # ── Detection heads for matching queries ─────────────────────────────
        layer_preds = []
        for layer_out in all_layer_outputs:
            match_out = layer_out[:, :self.n_slots, :].reshape(
                B, T, self.n_slots, -1
            )
            layer_preds.append(self.heads(match_out, B, T))

        result = {'layer_preds': layer_preds}

        # ── DN heads ─────────────────────────────────────────────────────────
        if gt is not None and self.training and S_dn > 0:
            dn_layer_preds = []
            for layer_out in all_layer_outputs:
                dn_out = layer_out[:, self.n_slots:, :].reshape(
                    B, T, S_dn, -1
                )
                dn_layer_preds.append(self.heads(dn_out, B, T))

            result['dn_layer_preds'] = dn_layer_preds
            result['pos_targets']    = pos_targets
            result['neg_targets']    = neg_targets
            result['S_dn']           = S_dn

        return result
