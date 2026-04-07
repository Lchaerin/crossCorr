"""
Real-time binaural audio engine for LIBERO simulation.

Design overview
---------------
BinauralAudioEngine manages one or more SoundSource objects.  Each source
loops a mono WAV file and is spatialised by convolving with a binaural HRTF
interpolated from a SOFA measurement database.

Spatial audio pipeline (per source, per block)
  1. Simulation step  → get object & camera poses
  2. compute_listener_relative_direction() → (azimuth, elevation, distance)
  3. HRTFInterpolator.interpolate()        → (hrir_l, hrir_r)
  4. OverlapAddConvolver.process()         → convolved stereo block
  5. sounddevice callback sums all sources → stereo output

HRTF interpolation strategy
----------------------------
HRTF measurements are discrete on the sphere.  We implement
**frequency-domain magnitude IDW** (inverse-distance weighting with
geodesic distance), which is the approach used in production spatial-audio
SDKs such as Google Resonance Audio and Steam Audio:

  - Find K nearest measured directions by geodesic (arc-length) distance.
  - Compute IDW weights  w_i = 1/d_i^2,  normalised.
  - Interpolate log-magnitude spectra  (perceptually motivated — ears are
    more sensitive to spectral envelope than to fine temporal structure).
  - Keep the nearest-neighbour's phase  (avoids phase-cancellation artefacts
    that linear phase averaging produces).

Upgrade path
  Minimum-phase decomposition + separate ITD interpolation (as in the
  IRCAM Spat framework or Apple Spatial Audio) would further improve quality,
  especially for large angular steps.  See MIN_PHASE_TODO comments.

Coordinate convention (az/el)
  azimuth  : 0° = front, 90° = left, −90° = right, ±180° = back
  elevation: 0° = horizon, 90° = directly above, −90° = directly below

Requirements
  pip install sounddevice soundfile netCDF4 scipy numpy
"""

from __future__ import annotations

import threading
import netCDF4 as nc
import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sph_to_cart(az_deg, el_deg) -> np.ndarray:
    """Spherical (degrees) → unit Cartesian.  Inputs may be scalars or arrays."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    if az.ndim == 0:
        return np.array([float(x), float(y), float(z)])
    return np.stack([x, y, z], axis=-1)


def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 1))))


def compute_listener_relative_direction(
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    obj_pos: np.ndarray,
) -> tuple[float, float, float]:
    """
    Convert a world-frame object position to the listener's head-frame
    spherical coordinates.

    MuJoCo camera matrix (cam_xmat) column layout
      col 0 → right direction   in world frame
      col 1 → up direction      in world frame
      col 2 → backward direction in world frame  (-forward)

    Parameters
    ----------
    cam_pos  : (3,) world position of the agentview camera (= ear position)
    cam_xmat : (9,) or (3,3) rotation matrix from sim.data.cam_xmat[cam_id]
    obj_pos  : (3,) world position of the sound source

    Returns
    -------
    azimuth_deg   : float  (0=front, 90=left, −90=right)
    elevation_deg : float  (0=horizon, 90=up)
    distance      : float  metres
    """
    mat     = np.asarray(cam_xmat, dtype=float).reshape(3, 3)
    right   =  mat[:, 0]
    up      =  mat[:, 1]
    forward = -mat[:, 2]   # MuJoCo camera looks along −Z of its local frame

    rel      = np.asarray(obj_pos, dtype=float) - np.asarray(cam_pos, dtype=float)
    distance = float(np.linalg.norm(rel))
    if distance < 1e-6:
        return 0.0, 0.0, 0.0

    x = float(np.dot(rel, right))      # positive → source is to the right
    y = float(np.dot(rel, up))         # positive → source is above
    z = float(np.dot(rel, forward))    # positive → source is in front

    # Azimuth: counter-clockwise from front when viewed from above.
    # Standard HRTF convention: left is positive.
    az = float(np.degrees(np.arctan2(-x, z)))
    el = float(np.degrees(np.arctan2(y, np.sqrt(x * x + z * z))))
    return az, el, distance


# ---------------------------------------------------------------------------
# SOFA loader
# ---------------------------------------------------------------------------

def load_sofa(
    sofa_path: str,
    target_sr: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load a SimpleFreeFieldHRIR SOFA file.

    Returns
    -------
    hrirs        : float64 ndarray  [M, 2, N]   (measurements × ears × samples)
    positions_deg: float64 ndarray  [M, 2]       (azimuth, elevation) in degrees
    sr           : int                            sample rate (after resampling)
    """
    ds = nc.Dataset(sofa_path, "r")
    hrirs    = np.array(ds.variables["Data.IR"][:],         dtype=np.float64)  # [M,2,N]
    pos      = np.array(ds.variables["SourcePosition"][:],  dtype=np.float64)  # [M,3]
    sofa_sr  = int(np.asarray(ds.variables["Data.SamplingRate"][:])[0])
    ds.close()

    if target_sr is not None and target_sr != sofa_sr:
        resampled = []
        for m in range(hrirs.shape[0]):
            l = scipy.signal.resample_poly(hrirs[m, 0], target_sr, sofa_sr)
            r = scipy.signal.resample_poly(hrirs[m, 1], target_sr, sofa_sr)
            resampled.append(np.stack([l, r]))
        hrirs = np.array(resampled)
        sr = target_sr
    else:
        sr = sofa_sr

    return hrirs, pos[:, :2], sr   # drop radius column


# ---------------------------------------------------------------------------
# HRTF interpolation
# ---------------------------------------------------------------------------

class HRTFInterpolator:
    """
    Interpolate binaural HRTFs at arbitrary sphere directions.

    Method
    ------
    For each query direction we find the K geographically nearest HRTF
    measurements (by geodesic / arc-length distance) and apply IDW weights
    in the **frequency domain**:

      1. Compute FFT of each K neighbour HRIR.
      2. Average log-magnitude spectra with IDW weights
         (log scale = perceptually uniform, matches dB sensitivity of hearing).
      3. Take phase from the single nearest measurement
         (avoids spectral smearing from linear phase averaging).
      4. Reconstruct interpolated HRIR via IFFT.

    MIN_PHASE_TODO
      A higher-quality extension: decompose each HRIR into its minimum-phase
      component (via cepstral windowing) and a pure-delay representing the
      ITD.  Interpolate minimum-phase components and ITD separately, then
      recombine.  This eliminates phase-cancellation artefacts entirely.
    """

    def __init__(
        self,
        hrirs: np.ndarray,        # [M, 2, N]
        positions_deg: np.ndarray, # [M, 2]  (azimuth, elevation)
        n_neighbors: int = 3,
    ):
        self.hrirs         = hrirs            # [M, 2, N]
        self.positions_deg = positions_deg    # [M, 2]
        self.n_neighbors   = n_neighbors
        self.hrir_len      = hrirs.shape[2]
        self.fft_size      = _next_pow2(self.hrir_len)

        # Precompute unit Cartesian directions for all measurements  [M, 3]
        self.directions = _sph_to_cart(positions_deg[:, 0], positions_deg[:, 1])

    # ------------------------------------------------------------------
    def interpolate(self, az_deg: float, el_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return interpolated (hrir_l, hrir_r), each ndarray of shape [N].

        Parameters
        ----------
        az_deg : azimuth in degrees   (0=front, 90=left, −90=right)
        el_deg : elevation in degrees (0=horizon, 90=up)
        """
        idx, dists = self._nearest(az_deg, el_deg)

        # Exact match → return as-is
        if dists[0] < 1e-6:
            return self.hrirs[idx[0], 0].copy(), self.hrirs[idx[0], 1].copy()

        # IDW weights:  w_i ∝ 1 / d_i²
        weights = 1.0 / (dists ** 2)
        weights /= weights.sum()

        hrir_l = self._interp_channel(idx, weights, ch=0)
        hrir_r = self._interp_channel(idx, weights, ch=1)
        return hrir_l, hrir_r

    # ------------------------------------------------------------------
    def _nearest(self, az_deg: float, el_deg: float):
        """Return (sorted_indices, angular_distances_rad) for K nearest."""
        q       = _sph_to_cart(az_deg, el_deg)
        cos_a   = np.clip(self.directions @ q, -1.0, 1.0)
        angles  = np.arccos(cos_a)                        # [M]
        idx     = np.argsort(angles)[: self.n_neighbors]
        return idx, angles[idx]

    def _interp_channel(self, idx: np.ndarray, weights: np.ndarray, ch: int) -> np.ndarray:
        """Frequency-domain magnitude interpolation for one ear channel."""
        hrirs_k  = self.hrirs[idx, ch, :]                             # [K, N]
        Hs       = np.fft.rfft(hrirs_k, n=self.fft_size, axis=1)      # [K, F]

        # Weighted average of log-magnitudes (dB scale)
        log_mags     = 20.0 * np.log10(np.abs(Hs) + 1e-10)            # [K, F]
        interp_logmag = (weights[:, None] * log_mags).sum(axis=0)      # [F]
        interp_mag    = 10.0 ** (interp_logmag / 20.0)

        # Phase from nearest neighbour (index 0 after argsort)
        phase = np.angle(Hs[0])

        H_interp = interp_mag * np.exp(1j * phase)
        hrir     = np.fft.irfft(H_interp, n=self.fft_size)[: self.hrir_len]
        return hrir.astype(np.float64)


# ---------------------------------------------------------------------------
# Overlap-add real-time convolver
# ---------------------------------------------------------------------------

class OverlapAddConvolver:
    """
    Single-channel real-time FIR convolution via the overlap-add algorithm.

    Supports lock-safe HRIR updates applied atomically at block boundaries
    (avoids mid-block discontinuities / clicks).
    """

    def __init__(self, hrir: np.ndarray, block_size: int):
        self.block_size = block_size
        self._lock    = threading.Lock()
        self._pending: np.ndarray | None = None
        self._init(hrir)

    # ------------------------------------------------------------------
    def _init(self, hrir: np.ndarray):
        self._hrir_len = len(hrir)
        self._fft_size = _next_pow2(self.block_size + self._hrir_len - 1)
        self._H        = np.fft.rfft(hrir, n=self._fft_size)
        self._overlap  = np.zeros(max(self._hrir_len - 1, 0), dtype=np.float64)

    def update_hrir(self, hrir: np.ndarray):
        """Schedule HRIR swap (applied at next block boundary)."""
        with self._lock:
            self._pending = hrir.copy()

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Convolve one block of mono input with the current HRIR.

        Parameters
        ----------
        x : ndarray  shape [block_size], dtype float64

        Returns
        -------
        ndarray  shape [block_size]
        """
        with self._lock:
            if self._pending is not None:
                self._init(self._pending)
                self._pending = None

        X      = np.fft.rfft(x, n=self._fft_size)
        y_full = np.fft.irfft(X * self._H, n=self._fft_size)

        out_len = self.block_size + self._hrir_len - 1
        y_full  = y_full[:out_len]

        out    = y_full[: self.block_size].copy()
        ov_len = min(len(self._overlap), self.block_size)
        out[:ov_len] += self._overlap[:ov_len]
        self._overlap = y_full[self.block_size :]
        return out


# ---------------------------------------------------------------------------
# Sound source
# ---------------------------------------------------------------------------

class SoundSource:
    """
    A single spatialized looping mono audio source.

    The source reads a WAV file (resampling if needed), loops it indefinitely,
    and applies left/right HRTF convolution each block.
    """

    def __init__(
        self,
        name: str,
        audio_path: str,
        sr: int,
        block_size: int,
        loop: bool = True,
    ):
        self.name       = name
        self.loop       = loop
        self.block_size = block_size

        # Load WAV → mono float64
        raw, orig_sr = sf.read(audio_path, always_2d=True)
        mono = raw.mean(axis=1)
        if orig_sr != sr:
            mono = scipy.signal.resample_poly(mono, sr, orig_sr)
        self.audio = mono.astype(np.float64)
        self._pos  = 0

        # Start with a Dirac (identity convolution) so audio is heard immediately
        dirac       = np.zeros(128, dtype=np.float64)
        dirac[0]    = 1.0
        self._conv_l = OverlapAddConvolver(dirac, block_size)
        self._conv_r = OverlapAddConvolver(dirac, block_size)
        self._gain      = 1.0
        self._gain_lock = threading.Lock()

    # ------------------------------------------------------------------
    def update_hrtf(
        self,
        hrir_l: np.ndarray,
        hrir_r: np.ndarray,
        gain: float = 1.0,
    ):
        """Thread-safe update of HRIRs and amplitude gain."""
        self._conv_l.update_hrir(hrir_l)
        self._conv_r.update_hrir(hrir_r)
        with self._gain_lock:
            self._gain = float(gain)

    def _read_block(self) -> np.ndarray:
        n   = len(self.audio)
        end = self._pos + self.block_size
        if end <= n:
            block = self.audio[self._pos : end].copy()
        elif self.loop:
            tail  = self.audio[self._pos :]
            head  = self.audio[: end - n]
            block = np.concatenate([tail, head])
        else:
            block = np.zeros(self.block_size, dtype=np.float64)
            rem   = n - self._pos
            if rem > 0:
                block[:rem] = self.audio[self._pos :]
        self._pos = (end % n) if self.loop else min(end, n)
        return block

    def process(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (left_block, right_block), each shape [block_size]."""
        block = self._read_block()
        with self._gain_lock:
            g = self._gain
        return self._conv_l.process(block) * g, self._conv_r.process(block) * g


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class BinauralAudioEngine:
    """
    Real-time binaural audio engine supporting multiple simultaneous sources.

    Typical usage in a simulation loop
    -----------------------------------
    ::
        # Setup (once)
        engine = BinauralAudioEngine(
            hrtf_path  = "audio_generation/hrtf/p0001.sofa",
            sample_rate = 44100,
        )
        engine.add_source("alphabet_soup_1", "audio_generation/sound/136.wav")
        engine.start()

        # Each simulation step
        az, el, dist = compute_listener_relative_direction(cam_pos, cam_xmat, obj_pos)
        engine.update_source_position("alphabet_soup_1", az, el, dist)

        # End of episode
        audio = engine.get_recorded_audio()   # [N_samples, 2] float32
        engine.clear_recording()

        # Teardown
        engine.stop()

    Extending to multiple / different objects
    -----------------------------------------
    Call add_source() once per object.  Pass a dict mapping object names to WAV
    paths, e.g.::

        AUDIO_SOURCES = {
            "alphabet_soup_1": "sound/136.wav",
            "basket_1":        "sound/ambient.wav",
        }
        for name, path in AUDIO_SOURCES.items():
            engine.add_source(name, path)

    Then in the loop::

        for name, obj_pos in object_positions.items():
            az, el, dist = compute_listener_relative_direction(cam_pos, cam_xmat, obj_pos)
            engine.update_source_position(name, az, el, dist)
    """

    def __init__(
        self,
        hrtf_path: str,
        sample_rate: int  = 44100,
        block_size: int   = 512,
        n_hrtf_neighbors: int = 3,
        noise_level: float = 0.003,
    ):
        self.sr          = sample_rate
        self.block_size  = block_size
        self.noise_level = noise_level   # RMS amplitude of additive white noise

        hrirs, positions, _ = load_sofa(hrtf_path, target_sr=sample_rate)
        self._interpolator  = HRTFInterpolator(hrirs, positions,
                                                n_neighbors=n_hrtf_neighbors)

        self.sources: dict[str, SoundSource] = {}
        self._stream: sd.OutputStream | None = None
        self._rec_buf:  list[np.ndarray]     = []
        self._rec_lock  = threading.Lock()

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def add_source(self, name: str, audio_path: str, loop: bool = True):
        """Register a named sound source backed by a WAV file."""
        self.sources[name] = SoundSource(
            name, audio_path, self.sr, self.block_size, loop
        )

    def update_source_position(
        self,
        name: str,
        az_deg: float,
        el_deg: float,
        distance: float,
    ):
        """
        Spatialise source *name* at (az_deg, el_deg, distance).

        Gain follows the inverse-distance law: gain = 1 / max(distance, 0.05).
        Unknown source names are silently ignored (safe to call for all objects
        even when only a subset have registered audio).
        """
        if name not in self.sources:
            return
        hrir_l, hrir_r = self._interpolator.interpolate(az_deg, el_deg)
        gain = 1.0 / max(float(distance), 0.05)
        self.sources[name].update_hrtf(hrir_l, hrir_r, gain)

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def start(self):
        """Open the audio output stream and begin playback."""
        self._stream = sd.OutputStream(
            samplerate = self.sr,
            channels   = 2,
            blocksize  = self.block_size,
            dtype      = "float32",
            callback   = self._callback,
        )
        self._stream.start()

    def stop(self):
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        out_l = np.zeros(frames, dtype=np.float64)
        out_r = np.zeros(frames, dtype=np.float64)
        for src in self.sources.values():
            l, r = src.process()
            out_l += l[:frames]
            out_r += r[:frames]
        # Soft-clip to prevent hard clipping when sources overlap
        out_l = np.tanh(out_l)
        out_r = np.tanh(out_r)
        # Additive white noise
        if self.noise_level > 0.0:
            out_l += np.random.normal(0.0, self.noise_level, frames)
            out_r += np.random.normal(0.0, self.noise_level, frames)
        out   = np.stack([out_l, out_r], axis=1).astype(np.float32)
        outdata[:] = out
        with self._rec_lock:
            self._rec_buf.append(out.copy())

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def get_recorded_audio(self) -> np.ndarray:
        """Return audio recorded since last clear_recording() as [N, 2] float32."""
        with self._rec_lock:
            if not self._rec_buf:
                return np.zeros((0, 2), dtype=np.float32)
            return np.concatenate(self._rec_buf, axis=0)

    def clear_recording(self):
        """Discard the current recording buffer (call at episode start)."""
        with self._rec_lock:
            self._rec_buf = []
