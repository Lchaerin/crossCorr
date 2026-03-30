#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Encoder
==================
Input:  [B, 5, 64, T]   5-channel feature tensor from AudioPreprocessor
        hrtf_ch [B, 64, 32]   fixed-size HRTF az×el heatmap
Output: (multi_scale_feats, enc_out)
  multi_scale_feats : list of [B, T, 256] tensors (7 candidates total)
  enc_out           : [B, T, 256]

Changes from v3 original
------------------------
  - BatchNorm → GroupNorm throughout (stable for small batches and streaming)
  - HRTFProjection: input is now [B, 64, 32] (fixed size), output [B, d_model]
    applied as a broadcast additive bias over the temporal axis (not temporal conv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Causal Conv2d
# =============================================================================

class CausalConv2d(nn.Module):
    """2-D convolution that is causal in time (last dim) and symmetric in freq."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size=(3, 3), freq_stride: int = 1):
        super().__init__()
        freq_k, time_k = kernel_size
        self.time_pad = time_k - 1
        self.freq_pad = freq_k // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size = (freq_k, time_k),
            stride      = (freq_stride, 1),
            padding     = 0,
            bias        = False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.time_pad, 0, self.freq_pad, self.freq_pad))
        return self.conv(x)


# =============================================================================
# ConvBlock — GroupNorm instead of BatchNorm2d
# =============================================================================

class ConvBlock(nn.Module):
    """CausalConv2d + GroupNorm + GELU with optional freq downsampling."""

    def __init__(self, in_ch: int, out_ch: int, freq_stride: int = 2):
        super().__init__()
        self.conv = CausalConv2d(in_ch, out_ch, kernel_size=(3, 3),
                                 freq_stride=freq_stride)
        self.gn   = nn.GroupNorm(8, out_ch)   # 8 groups; works for 64/128/256
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


# =============================================================================
# SEBlock
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation on [B, C, F, T]."""

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
        B, C = x.shape[:2]
        s     = x.mean(dim=(2, 3))
        scale = self.fc(s).view(B, C, 1, 1)
        return x * scale


# =============================================================================
# CausalConformerBlock — GroupNorm instead of BatchNorm1d in depthwise conv
# =============================================================================

class CausalConformerBlock(nn.Module):
    """Causal Conformer block on [B, T, d].

    Order: Feed-Forward → Self-Attention (causal) → Conv (causal DW) → Feed-Forward
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_dim: int = 512, conv_kernel: int = 31,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        self.conv_norm = nn.LayerNorm(d_model)
        pad_left       = conv_kernel - 1
        self.conv_pw1  = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.conv_dw   = nn.Conv1d(
            d_model, d_model,
            kernel_size = conv_kernel,
            padding     = 0,
            groups      = d_model,
        )
        self.conv_pad  = pad_left
        self.conv_gn   = nn.GroupNorm(32, d_model)   # replaces BatchNorm1d
        self.conv_pw2  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_drop = nn.Dropout(dropout)

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
        mask = torch.ones(T, T, dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]

        x = x + 0.5 * self.ffn1(x)

        x_norm    = self.attn_norm(x)
        mask      = self._causal_attn_mask(T, x.device)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=mask, need_weights=False)
        x = x + self.attn_drop(attn_out)

        x_norm  = self.conv_norm(x)
        xc      = x_norm.transpose(1, 2)
        xc      = self.conv_pw1(xc)
        xc, xg  = xc.chunk(2, dim=1)
        xc      = xc * xg.sigmoid()
        xc      = F.pad(xc, (self.conv_pad, 0))
        xc      = self.conv_dw(xc)
        xc      = self.conv_gn(xc)
        xc      = F.silu(xc)
        xc      = self.conv_pw2(xc)
        xc      = xc.transpose(1, 2)
        x       = x + self.conv_drop(xc)

        x = x + 0.5 * self.ffn2(x)

        return self.final_norm(x)


# =============================================================================
# CausalBiFPN — GroupNorm instead of BatchNorm2d
# =============================================================================

class CausalBiFPN(nn.Module):
    """Causal Bidirectional Feature Pyramid Network."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        d = d_model

        self.td_conv54 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.td_conv43 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.td_gn54   = nn.GroupNorm(8, d)
        self.td_gn43   = nn.GroupNorm(8, d)

        self.bu_conv34 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.bu_conv45 = CausalConv2d(d, d, kernel_size=(3, 3))
        self.bu_gn34   = nn.GroupNorm(8, d)
        self.bu_gn45   = nn.GroupNorm(8, d)

        self.w_td = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(2)])
        self.w_bu = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(2)])

    @staticmethod
    def _fast_norm(feats, weights):
        w = F.relu(weights)
        w = w / (w.sum() + 1e-4)
        return sum(w[i] * feats[i] for i in range(len(feats)))

    def forward(self, feats: list) -> list:
        P3, P4, P5 = feats

        P4_td = self._fast_norm(
            [P4, F.interpolate(P5, size=(P4.shape[2], P4.shape[3]), mode='nearest')],
            self.w_td[0]
        )
        P4_td = F.gelu(self.td_gn54(self.td_conv54(P4_td)))

        P3_td = self._fast_norm(
            [P3, F.interpolate(P4_td, size=(P3.shape[2], P3.shape[3]), mode='nearest')],
            self.w_td[1]
        )
        P3_td = F.gelu(self.td_gn43(self.td_conv43(P3_td)))

        P4_bu = self._fast_norm(
            [P4_td, F.adaptive_avg_pool2d(P3_td, (P4.shape[2], P4.shape[3]))],
            self.w_bu[0]
        )
        P4_bu = F.gelu(self.bu_gn34(self.bu_conv34(P4_bu)))

        P5_bu = self._fast_norm(
            [P5, F.adaptive_avg_pool2d(P4_bu, (P5.shape[2], P5.shape[3]))],
            self.w_bu[1]
        )
        P5_bu = F.gelu(self.bu_gn45(self.bu_conv45(P5_bu)))

        return [P3_td, P4_bu, P5_bu]


# =============================================================================
# HRTF Projection — now takes fixed [B, 64, 32] → global [B, d_model]
# =============================================================================

class HRTFProjection(nn.Module):
    """Maps fixed-size HRTF az×el heatmap [B, az_bins, el_bins] → [B, d_model].

    Outputs a single global spatial embedding that is broadcast-added over the
    temporal axis in SLEDEncoder.forward.  No more variable T_stft dependency.
    """

    def __init__(self, az_bins: int = 64, el_bins: int = 32, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(1),                          # [B, az*el]
            nn.Linear(az_bins * el_bins, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, ch5: torch.Tensor) -> torch.Tensor:
        # ch5: [B, az_bins, el_bins]
        return self.proj(ch5)   # [B, d_model]


# =============================================================================
# SLED Encoder
# =============================================================================

class SLEDEncoder(nn.Module):
    """Full SLED v3 encoder."""

    def __init__(self, in_channels: int = 5, d_model: int = 256,
                 n_bifpn: int = 2, n_conformer: int = 4,
                 use_hrtf_corr: bool = True):
        super().__init__()

        self.stem3 = ConvBlock(in_channels, 64,  freq_stride=2)
        self.stem4 = ConvBlock(64,          128, freq_stride=2)
        self.stem5 = ConvBlock(128,         256, freq_stride=2)
        self.se5   = SEBlock(256)

        self.proj3 = nn.Conv2d(64,  d_model, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(128, d_model, kernel_size=1, bias=False)
        self.proj5 = nn.Conv2d(256, d_model, kernel_size=1, bias=False)

        self.bifpns = nn.ModuleList([CausalBiFPN(d_model) for _ in range(n_bifpn)])

        freq_dim_p5    = 8
        self.freq_proj = nn.Conv1d(d_model * freq_dim_p5, d_model,
                                   kernel_size=1, bias=False)
        self.freq_gn   = nn.GroupNorm(32, d_model)   # replaces BatchNorm1d

        self.conformers = nn.ModuleList([
            CausalConformerBlock(d_model) for _ in range(n_conformer)
        ])

        self.hrtf_proj = HRTFProjection(az_bins=64, el_bins=32, d_model=d_model) \
            if use_hrtf_corr else None
        self.d_model   = d_model

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, feat: torch.Tensor,
                hrtf_ch: torch.Tensor | None = None):
        """
        Parameters
        ----------
        feat    : [B, 5, 64, T]
        hrtf_ch : [B, 64, 32]  fixed-size HRTF spatial map, or None

        Returns
        -------
        multi_scale_feats : list of 7 × [B, T, d]
        enc_out           : [B, T, d]
        """
        P3 = self.stem3(feat)
        P4 = self.stem4(P3)
        P5 = self.stem5(P4)
        P5 = self.se5(P5)

        P3 = self.proj3(P3)
        P4 = self.proj4(P4)
        P5 = self.proj5(P5)

        feats = [P3, P4, P5]
        for bifpn in self.bifpns:
            feats = bifpn(feats)
        P3, P4, P5 = feats

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

        B, d, F5, T = P5.shape
        p5_flat = P5.reshape(B, d * F5, T)
        p5_enc  = F.gelu(self.freq_gn(self.freq_proj(p5_flat)))
        x       = p5_enc.permute(0, 2, 1)   # [B, T, d]

        for conformer in self.conformers:
            x = conformer(x)

        # HRTF: global spatial bias broadcast over temporal axis
        if hrtf_ch is not None:
            hrtf_feat = self.hrtf_proj(hrtf_ch)   # [B, d]
            x = x + hrtf_feat.unsqueeze(1)          # [B, T, d]

        enc_out = x
        ms_feats.append(enc_out)   # 7th candidate

        return ms_feats, enc_out
