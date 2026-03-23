#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Encoder
==================
Input:  [B, 5, 64, T]   5-channel feature tensor from AudioPreprocessor
        hrtf_ch [B, 64, T]   HRTF az×el heatmap (channel 5, separate branch)
Output: (multi_scale_feats, enc_out)
  multi_scale_feats : list of [B, T, 256] tensors (7 candidates total)
  enc_out           : [B, T, 256]

Architecture
------------
  ConvBlock(5→64,   freq_stride=2)  → P3 [B, 64, 32, T]
  ConvBlock(64→128, freq_stride=2)  → P4 [B,128, 16, T]
  ConvBlock(128→256,freq_stride=2)  → P5 [B,256,  8, T]
  SEBlock(256) on P5
  Project P3,P4,P5 to 256 channels (1×1 Conv2d)
  CausalBiFPN × 2
  Flatten P5: [B,256,8,T] → [B,256*8,T] → Conv1d(256*8,256,1) → [B,256,T]
  Permute → [B, T, 256]
  CausalConformerBlock × 4

Multi-scale features (7 candidates for MixedQuerySelector):
  P3 upper half / lower half freq-pooled   → 2 × [B,T,256]
  P4 upper half / lower half freq-pooled   → 2 × [B,T,256]
  P5 upper half / lower half freq-pooled   → 2 × [B,T,256]
  enc_out                                  → 1 × [B,T,256]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Causal Conv2d
# =============================================================================

class CausalConv2d(nn.Module):
    """2-D convolution that is causal in time (last dim) and symmetric in freq.

    Parameters
    ----------
    kernel_size : (freq_k, time_k)
    freq_stride : stride in frequency dimension
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size=(3, 3), freq_stride: int = 1):
        super().__init__()
        freq_k, time_k = kernel_size

        # Causal padding in time: pad (time_k-1) on the left, 0 on the right
        self.time_pad = time_k - 1
        # Symmetric padding in frequency
        self.freq_pad = freq_k // 2

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size = (freq_k, time_k),
            stride      = (freq_stride, 1),
            padding     = 0,    # manual padding below
            bias        = False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        # Pad: (left_T, right_T, top_F, bottom_F)
        x = F.pad(x, (self.time_pad, 0, self.freq_pad, self.freq_pad))
        return self.conv(x)


# =============================================================================
# ConvBlock
# =============================================================================

class ConvBlock(nn.Module):
    """CausalConv2d + BatchNorm2d + GELU with optional freq downsampling."""

    def __init__(self, in_ch: int, out_ch: int, freq_stride: int = 2):
        super().__init__()
        self.conv = CausalConv2d(
            in_ch, out_ch,
            kernel_size  = (3, 3),
            freq_stride  = freq_stride,
        )
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# =============================================================================
# SEBlock
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block on the channel dimension.

    Operates on [B, C, F, T] input — squeezes over F and T.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        B, C = x.shape[:2]
        s = x.mean(dim=(2, 3))                 # [B, C]
        scale = self.fc(s).view(B, C, 1, 1)    # [B, C, 1, 1]
        return x * scale


# =============================================================================
# CausalConformerBlock
# =============================================================================

class CausalConformerBlock(nn.Module):
    """Causal Conformer block on [B, T, d].

    Order: Feed-Forward → Self-Attention (causal) → Conv (causal DW) → Feed-Forward
    Half-residual scaling on feed-forward sub-layers (0.5 × FFN + input).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_dim: int = 512, conv_kernel: int = 31,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Feed-Forward 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # Causal Multi-Head Self-Attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        # Causal Depthwise Conv
        self.conv_norm = nn.LayerNorm(d_model)
        pad_left       = conv_kernel - 1
        # pointwise expand → depthwise (causal left-pad) → pointwise project
        self.conv_pw1  = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.conv_dw   = nn.Conv1d(
            d_model, d_model,
            kernel_size = conv_kernel,
            padding     = 0,       # manual left-pad
            groups      = d_model,
        )
        self.conv_pad  = pad_left
        self.conv_bn   = nn.BatchNorm1d(d_model)
        self.conv_pw2  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_drop = nn.Dropout(dropout)

        # Feed-Forward 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def _causal_attn_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = blocked), shape [T, T]."""
        mask = torch.ones(T, T, dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d]
        T = x.shape[1]

        # FF1 (half residual)
        x = x + 0.5 * self.ffn1(x)

        # Causal self-attention
        x_norm    = self.attn_norm(x)
        mask      = self._causal_attn_mask(T, x.device)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=mask, need_weights=False)
        x = x + self.attn_drop(attn_out)

        # Causal depthwise conv
        x_norm  = self.conv_norm(x)                   # [B, T, d]
        xc      = x_norm.transpose(1, 2)              # [B, d, T]
        xc      = self.conv_pw1(xc)                   # [B, 2d, T]
        xc, xg  = xc.chunk(2, dim=1)                  # GLU gate
        xc      = xc * xg.sigmoid()                   # [B, d, T]
        xc      = F.pad(xc, (self.conv_pad, 0))       # causal left-pad
        xc      = self.conv_dw(xc)                    # [B, d, T]
        xc      = self.conv_bn(xc)
        xc      = F.silu(xc)
        xc      = self.conv_pw2(xc)                   # [B, d, T]
        xc      = xc.transpose(1, 2)                  # [B, T, d]
        x       = x + self.conv_drop(xc)

        # FF2 (half residual)
        x = x + 0.5 * self.ffn2(x)

        return self.final_norm(x)


# =============================================================================
# CausalBiFPN
# =============================================================================

class CausalBiFPN(nn.Module):
    """Causal Bidirectional Feature Pyramid Network.

    Operates on three scales in [B, d, F_i, T] format:
        P3 [B, d, F3, T]  (finest)
        P4 [B, d, F4, T]
        P5 [B, d, F5, T]  (coarsest)

    Fuses top-down then bottom-up; all convolutions are CausalConv2d.
    """

    def __init__(self, d_model: int = 256, n_levels: int = 3):
        super().__init__()
        d = d_model

        # Top-down: P5 → P4 → P3
        self.td_conv54 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.td_conv43 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.td_bn54   = nn.BatchNorm2d(d)
        self.td_bn43   = nn.BatchNorm2d(d)

        # Bottom-up: P3 → P4 → P5
        self.bu_conv34 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.bu_conv45 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.bu_bn34   = nn.BatchNorm2d(d)
        self.bu_bn45   = nn.BatchNorm2d(d)

        # Learnable fast-normalisation weights (per-fusion node)
        # 3 fusion nodes, each with 2 inputs
        self.w_td = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(2)  # td54, td43
        ])
        self.w_bu = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(2)  # bu34, bu45
        ])

    @staticmethod
    def _fast_norm(feats, weights):
        """Fast-normalised weighted fusion."""
        w = F.relu(weights)
        w = w / (w.sum() + 1e-4)
        return sum(w[i] * feats[i] for i in range(len(feats)))

    def forward(self, feats: list) -> list:
        """
        Parameters
        ----------
        feats : [P3, P4, P5]  each [B, d, F_i, T]

        Returns
        -------
        [P3', P4', P5']  same shapes
        """
        P3, P4, P5 = feats

        # ── Top-down ─────────────────────────────────────────────────────────
        # Upsample P5 freq to P4 freq
        P4_td = self._fast_norm(
            [P4, F.interpolate(P5, size=(P4.shape[2], P4.shape[3]),
                               mode='nearest')],
            self.w_td[0]
        )
        P4_td = F.gelu(self.td_bn54(self.td_conv54(P4_td)))

        # Upsample P4_td freq to P3 freq
        P3_td = self._fast_norm(
            [P3, F.interpolate(P4_td, size=(P3.shape[2], P3.shape[3]),
                               mode='nearest')],
            self.w_td[1]
        )
        P3_td = F.gelu(self.td_bn43(self.td_conv43(P3_td)))

        # ── Bottom-up ─────────────────────────────────────────────────────────
        # Downsample P3_td freq to P4 freq
        P4_bu = self._fast_norm(
            [P4_td, F.adaptive_avg_pool2d(P3_td, (P4.shape[2], P4.shape[3]))],
            self.w_bu[0]
        )
        P4_bu = F.gelu(self.bu_bn34(self.bu_conv34(P4_bu)))

        # Downsample P4_bu freq to P5 freq
        P5_bu = self._fast_norm(
            [P5, F.adaptive_avg_pool2d(P4_bu, (P5.shape[2], P5.shape[3]))],
            self.w_bu[1]
        )
        P5_bu = F.gelu(self.bu_bn45(self.bu_conv45(P5_bu)))

        return [P3_td, P4_bu, P5_bu]


# =============================================================================
# HRTF Projection
# =============================================================================

class HRTFProjection(nn.Module):
    """Projects the HRTF az×el heatmap [B, az_bins, T] → [B, T, d_model].

    Treats azimuth bins as channels and collapses them via Conv1d.
    The elevation dimension (T_stft) aligns with the temporal axis of enc_out.
    """

    def __init__(self, az_bins: int = 64, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(az_bins, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, ch5: torch.Tensor) -> torch.Tensor:
        # ch5: [B, 64_az, T]
        x = self.proj(ch5)          # [B, d_model, T]
        return x.permute(0, 2, 1)   # [B, T, d_model]


# =============================================================================
# SLED Encoder
# =============================================================================

class SLEDEncoder(nn.Module):
    """Full SLED v3 encoder.

    Parameters
    ----------
    in_channels     : number of input feature channels (5, excluding ch5 HRTF)
    d_model         : feature dimension for BiFPN + Conformer (256)
    n_bifpn         : number of BiFPN iterations (2)
    n_conformer     : number of CausalConformerBlock layers (4)
    """

    def __init__(self, in_channels: int = 5, d_model: int = 256,
                 n_bifpn: int = 2, n_conformer: int = 4):
        super().__init__()

        # ── Conv stem ────────────────────────────────────────────────────────
        # P3: [B, in_channels, 64, T] → [B,  64, 32, T]
        # P4: [B,  64, 32, T] → [B, 128, 16, T]
        # P5: [B, 128, 16, T] → [B, 256,  8, T]
        self.stem3 = ConvBlock(in_channels, 64,  freq_stride=2)
        self.stem4 = ConvBlock(64,          128, freq_stride=2)
        self.stem5 = ConvBlock(128,         256, freq_stride=2)
        self.se5   = SEBlock(256)

        # Project all scales to d_model channels
        self.proj3 = nn.Conv2d(64,  d_model, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(128, d_model, kernel_size=1, bias=False)
        self.proj5 = nn.Conv2d(256, d_model, kernel_size=1, bias=False)

        # ── CausalBiFPN ──────────────────────────────────────────────────────
        self.bifpns = nn.ModuleList([CausalBiFPN(d_model) for _ in range(n_bifpn)])

        # ── Temporal flatten + project ────────────────────────────────────────
        # P5 after BiFPN: [B, d, 8, T] → flatten freq → [B, d*8, T]
        # Then Conv1d to [B, d, T]
        freq_dim_p5 = 8  # 64 / 2 / 2 / 2
        self.freq_proj = nn.Conv1d(d_model * freq_dim_p5, d_model,
                                   kernel_size=1, bias=False)
        self.freq_bn   = nn.BatchNorm1d(d_model)

        # ── Causal Conformer ─────────────────────────────────────────────────
        self.conformers = nn.ModuleList([
            CausalConformerBlock(d_model) for _ in range(n_conformer)
        ])

        # ── HRTF projection (separate branch for ch5) ────────────────────────
        self.hrtf_proj = HRTFProjection(az_bins=64, d_model=d_model)

        self.d_model = d_model

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, feat: torch.Tensor,
                hrtf_ch: torch.Tensor | None = None):
        """
        Parameters
        ----------
        feat    : [B, 5, 64, T]
        hrtf_ch : [B, 64, T]  HRTF az×el heatmap, or None (precomputed mode)

        Returns
        -------
        multi_scale_feats : list of 7 × [B, T, 256]
        enc_out           : [B, T, 256]
        """
        # ── Stem ─────────────────────────────────────────────────────────────
        P3 = self.stem3(feat)   # [B,  64, 32, T]
        P4 = self.stem4(P3)     # [B, 128, 16, T]
        P5 = self.stem5(P4)     # [B, 256,  8, T]
        P5 = self.se5(P5)

        # Project to d_model
        P3 = self.proj3(P3)     # [B, 256, 32, T]
        P4 = self.proj4(P4)     # [B, 256, 16, T]
        P5 = self.proj5(P5)     # [B, 256,  8, T]

        # ── BiFPN ────────────────────────────────────────────────────────────
        feats = [P3, P4, P5]
        for bifpn in self.bifpns:
            feats = bifpn(feats)
        P3, P4, P5 = feats

        # ── Multi-scale candidates (6 sub-band features) ─────────────────────
        # For each scale [B, d, F, T]: split F in half → avg-pool each half → [B,T,d]
        def _subband_pool(p: torch.Tensor):
            """[B,d,F,T] → two [B,T,d] tensors (lower / upper half of freq)"""
            F_dim = p.shape[2]
            mid   = F_dim // 2
            lo    = p[:, :, :mid, :].mean(dim=2).permute(0, 2, 1)  # [B,T,d]
            hi    = p[:, :, mid:, :].mean(dim=2).permute(0, 2, 1)
            return lo, hi

        ms_feats = []
        for px in (P3, P4, P5):
            lo, hi = _subband_pool(px)
            ms_feats.extend([lo, hi])   # 6 candidates so far

        # ── Temporal encoder ─────────────────────────────────────────────────
        B, d, F5, T = P5.shape
        p5_flat = P5.reshape(B, d * F5, T)          # [B, d*8, T]
        p5_enc  = F.gelu(self.freq_bn(self.freq_proj(p5_flat)))  # [B, d, T]
        x       = p5_enc.permute(0, 2, 1)           # [B, T, d]

        for conformer in self.conformers:
            x = conformer(x)

        # ── HRTF residual fusion ──────────────────────────────────────────────
        if hrtf_ch is not None:
            # Align T in case of minor length mismatch
            T_enc = x.shape[1]
            hrtf_feat = self.hrtf_proj(hrtf_ch)         # [B, T_hrtf, d]
            hrtf_feat = hrtf_feat[:, :T_enc, :]          # trim to enc T
            x = x + hrtf_feat

        enc_out = x   # [B, T, 256]

        # Add enc_out as the 7th candidate
        ms_feats.append(enc_out)

        return ms_feats, enc_out
