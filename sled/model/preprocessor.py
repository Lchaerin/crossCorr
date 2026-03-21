#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Audio Preprocessor
==============================
Converts a stereo waveform [B, 2, N] → 6-channel feature tensor [B, 6, 64, T].

Channels
--------
  0  L-mel       log mel spectrogram, left channel
  1  R-mel       log mel spectrogram, right channel
  2  ILD         per-mel-band inter-level difference (dB)
  3  sin(IPD)    per-mel-band inter-phase difference — sine component
  4  cos(IPD)    per-mel-band inter-phase difference — cosine component
  5  HRTF-corr   normalised HRTF cross-correlation heatmap (N_DIR → 64 az-bins)

STFT parameters
---------------
  n_fft=2048, hop_length=960, win_length=2048, window=hann
  n_mels=64, fmin=20, fmax=16000, sr=48000
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# =============================================================================
# Helper: build mel filterbank (triangular, HTK-scale)
# =============================================================================

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(n_fft: int, n_mels: int, sr: int,
                           fmin: float, fmax: float) -> np.ndarray:
    """Build a mel filterbank matrix [n_mels, n_fft//2+1].

    Uses triangular filters on the mel scale (HTK convention).
    """
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)  # [n_freqs]

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = np.array([_mel_to_hz(m) for m in mel_points])  # [n_mels+2]

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_low  = hz_points[m - 1]
        f_ctr  = hz_points[m]
        f_high = hz_points[m + 1]

        lower = (fft_freqs - f_low)  / max(f_ctr  - f_low,  1e-8)
        upper = (f_high - fft_freqs) / max(f_high - f_ctr,   1e-8)
        fb[m - 1, :] = np.maximum(0.0, np.minimum(lower, upper))

    return fb  # [n_mels, n_freqs]


# =============================================================================
# Preprocessor module
# =============================================================================

class AudioPreprocessor(nn.Module):
    """Converts stereo waveform → 6-channel feature tensor.

    Input  : [B, 2, N_samples]   stereo waveform at 48 kHz
    Output : [B, 6, 64, T]       feature tensor
    """

    def __init__(self, sofa_path: str, sr: int = 48_000,
                 n_fft: int = 2048, hop_length: int = 960,
                 n_mels: int = 64, fmin: float = 20.0, fmax: float = 16_000.0):
        super().__init__()
        self.sr         = sr
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_mels     = n_mels
        self.fmin       = fmin
        self.fmax       = fmax

        # ── Hann window ───────────────────────────────────────────────────────
        win = torch.hann_window(n_fft)
        self.register_buffer('window', win)

        # ── Mel filterbank [n_mels, n_fft//2+1] ──────────────────────────────
        mel_fb = _build_mel_filterbank(n_fft, n_mels, sr, fmin, fmax)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))  # [64, 1025]

        # ── HRTF buffers ──────────────────────────────────────────────────────
        self._load_hrtf_buffers(sofa_path)

    # ─────────────────────────────────────────────────────────────────────────

    def _load_hrtf_buffers(self, sofa_path: str):
        """Load SOFA HRTF and pre-compute frequency-domain cross-correlation
        weights, then register them as non-trainable buffers.

        Buffers registered
        ------------------
        W_real      [N_DIR, F]   Re(HRTF_R * conj(HRTF_L))
        W_imag      [N_DIR, F]   Im(HRTF_R * conj(HRTF_L))
        norm_hr_sq  [N_DIR, F]   |HRTF_R|^2
        norm_hl_sq  [N_DIR, F]   |HRTF_L|^2
        az_bin_idx  [N_DIR]      azimuth bin (0..63) for each direction
        """
        # Load HRIR
        try:
            import netCDF4
            ds = netCDF4.Dataset(sofa_path, 'r')
            ir_data    = np.array(ds.variables['Data.IR'][:])         # (M, 2, N_ir)
            source_pos = np.array(ds.variables['SourcePosition'][:])  # (M, 3)
            ds.close()
        except Exception:
            import h5py
            with h5py.File(sofa_path, 'r') as f_:
                ir_data    = np.array(f_['Data.IR'])
                source_pos = np.array(f_['SourcePosition'])

        hrir_l_np = ir_data[:, 0, :]   # (M, N_ir)
        hrir_r_np = ir_data[:, 1, :]
        azimuths  = source_pos[:, 0]   # degrees, SOFA convention

        N_DIR = hrir_l_np.shape[0]
        F     = self.n_fft // 2 + 1

        # Pad / truncate IR to n_fft length
        n_ir = hrir_l_np.shape[1]
        if n_ir < self.n_fft:
            pad   = self.n_fft - n_ir
            hl_np = np.pad(hrir_l_np, ((0, 0), (0, pad)))
            hr_np = np.pad(hrir_r_np, ((0, 0), (0, pad)))
        else:
            hl_np = hrir_l_np[:, :self.n_fft]
            hr_np = hrir_r_np[:, :self.n_fft]

        # RFFT of HRIRs → [N_DIR, F] complex
        HRTF_L = np.fft.rfft(hl_np, n=self.n_fft, axis=-1)  # complex (M, F)
        HRTF_R = np.fft.rfft(hr_np, n=self.n_fft, axis=-1)

        cross = HRTF_R * np.conj(HRTF_L)  # R × L* → (M, F) complex

        W_real_np     = cross.real.astype(np.float32)
        W_imag_np     = cross.imag.astype(np.float32)
        norm_hr_sq_np = (np.abs(HRTF_R) ** 2).astype(np.float32)
        norm_hl_sq_np = (np.abs(HRTF_L) ** 2).astype(np.float32)

        # ── Azimuth → 64 bins (sorted CW: az_sled = -az_sofa mod 360) ────────
        az_sled = (-azimuths) % 360.0      # convert SOFA CCW → SLED CW
        # Map each direction to one of n_mels=64 bins uniformly covering [0, 360)
        az_bin_idx = np.floor(az_sled / 360.0 * self.n_mels).astype(np.int64)
        az_bin_idx = np.clip(az_bin_idx, 0, self.n_mels - 1)

        self.register_buffer('W_real',      torch.from_numpy(W_real_np))
        self.register_buffer('W_imag',      torch.from_numpy(W_imag_np))
        self.register_buffer('norm_hr_sq',  torch.from_numpy(norm_hr_sq_np))
        self.register_buffer('norm_hl_sq',  torch.from_numpy(norm_hl_sq_np))
        self.register_buffer('az_bin_idx',  torch.from_numpy(az_bin_idx))
        self._N_DIR = N_DIR

    # ─────────────────────────────────────────────────────────────────────────

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT of x: [B, N] → complex [B, F, T_stft].

        Uses torch.stft with return_complex=True.
        """
        B = x.shape[0]
        # torch.stft expects [..., N]; pad both sides by n_fft//2 so that
        # frame 0 is centred at sample 0 (same convention as librosa center=True)
        pad = self.n_fft // 2
        x_pad = F.pad(x, (pad, pad))
        stft = torch.stft(
            x_pad,
            n_fft          = self.n_fft,
            hop_length     = self.hop_length,
            win_length     = self.n_fft,
            window         = self.window,
            center         = False,
            return_complex = True,
        )   # [B, F, T_stft]
        return stft

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : [B, 2, N]  stereo float32 at self.sr

        Returns
        -------
        [B, 6, 64, T]  float32 feature tensor
        """
        B = waveform.shape[0]
        x_L = waveform[:, 0, :]   # [B, N]
        x_R = waveform[:, 1, :]

        # ── STFT ─────────────────────────────────────────────────────────────
        X_L = self._stft(x_L)   # [B, F, T_stft] complex
        X_R = self._stft(x_R)

        # Power spectra
        pow_L = X_L.abs().pow(2)   # [B, F, T_stft]
        pow_R = X_R.abs().pow(2)

        # ── Mel spectrogram (linear power → mel-band → log) ──────────────────
        # mel_fb: [n_mels, F] → matmul with pow [B, F, T] → [B, n_mels, T]
        mel_fb = self.mel_fb  # [64, F]

        mel_L_lin = torch.einsum('mf,bft->bmt', mel_fb, pow_L)  # [B,64,T]
        mel_R_lin = torch.einsum('mf,bft->bmt', mel_fb, pow_R)

        eps = 1e-8
        ch0 = (mel_L_lin + eps).log10() * 10.0  # L-mel (dB-ish log power)
        ch1 = (mel_R_lin + eps).log10() * 10.0  # R-mel

        # ── ILD per mel band ─────────────────────────────────────────────────
        # ILD = 10 * log10(mel_L / mel_R) [linear mel, before log]
        ch2 = 10.0 * (mel_L_lin / (mel_R_lin + eps) + eps).log10()  # [B,64,T]

        # ── IPD per mel band ─────────────────────────────────────────────────
        # CSD = X_L * conj(X_R): [B, F, T] complex
        csd_full = X_L * X_R.conj()

        # Average CSD phase over frequencies within each mel band
        # mel_fb[m, f] = weight for mel band m at frequency f
        # mel_csd[B, n_mels, T] = weighted sum of CSD over freq
        mel_csd_real = torch.einsum('mf,bft->bmt', mel_fb, csd_full.real)
        mel_csd_imag = torch.einsum('mf,bft->bmt', mel_fb, csd_full.imag)
        # Normalise by L2 norm to get unit-vector (cos, sin)
        mel_csd_norm = (mel_csd_real.pow(2) + mel_csd_imag.pow(2) + eps).sqrt()
        ch3 = mel_csd_imag / mel_csd_norm   # sin(IPD) [B, 64, T]
        ch4 = mel_csd_real / mel_csd_norm   # cos(IPD) [B, 64, T]

        # ── HRTF cross-correlation heatmap (channel 5) ────────────────────────
        T_stft = X_L.shape[-1]

        # CSD = X_L * conj(X_R): [B, F, T]   (real and imag parts)
        csd_r = csd_full.real   # [B, F, T]
        csd_i = csd_full.imag

        # W_real, W_imag: [N_DIR, F]
        # corr_unnorm[B, N_DIR, T] = W_real @ csd_r - W_imag @ csd_i
        # Use einsum: 'df,bft->bdt'
        corr_unnorm = (
            torch.einsum('df,bft->bdt', self.W_real, csd_r) -
            torch.einsum('df,bft->bdt', self.W_imag, csd_i)
        )  # [B, N_DIR, T]

        # norm_hr_sq @ |X_L|^2  → [B, N_DIR, T]
        norm1_sq = torch.einsum('df,bft->bdt', self.norm_hr_sq, pow_L)
        norm2_sq = torch.einsum('df,bft->bdt', self.norm_hl_sq, pow_R)
        corr = corr_unnorm / (norm1_sq * norm2_sq + 1e-8).sqrt()  # [B, N_DIR, T]

        # Pool N_DIR → 64 azimuth bins via scatter mean
        # az_bin_idx: [N_DIR]
        n_bins = self.n_mels   # 64
        idx    = self.az_bin_idx.view(1, -1, 1).expand(B, -1, T_stft)  # [B,N_DIR,T]
        hrtf_ch = torch.zeros(B, n_bins, T_stft,
                              dtype=corr.dtype, device=corr.device)
        count   = torch.zeros(B, n_bins, T_stft,
                              dtype=corr.dtype, device=corr.device)
        hrtf_ch.scatter_add_(1, idx, corr)
        count.scatter_add_(1, idx,
                           torch.ones_like(corr))
        hrtf_ch = hrtf_ch / (count + 1e-8)   # [B, 64, T]
        ch5 = hrtf_ch

        # ── Stack all 6 channels ─────────────────────────────────────────────
        # Each channel is [B, 64, T]
        out = torch.stack([ch0, ch1, ch2, ch3, ch4, ch5], dim=1)  # [B, 6, 64, T]
        return out
