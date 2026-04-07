#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Binaural Scene Synthesizer (v2: continuous positions + SRIR)
=======================================================================
Synthesizes binaural audio scenes for SLED training using:
  - Continuous random position sampling (az/el from continuous ranges)
  - HRTFInterpolator (binaural_engine.py) for frequency-domain IDW interpolation
    → HRTF subject randomly selected per scene from hrtf/p*.sofa (140 subjects)
  - TAU-SRIR room acoustics via W-channel (omnidirectional) convolution
    → room/condition randomly selected per scene from all 9 rooms
  - No separate noise (SRIR reverb tail naturally captures room noise)

Pipeline per source:
  1. Per scene: pick random HRTF subject (p*.sofa) + random SRIR room/condition
  2. Sample (az, el) from continuous space: az ∈ [−180, 180]°, el ∈ [−45, 45]°
  3. Interpolate HRTF at exact (az, el)  → hrir_l, hrir_r  [exact direction]
  4. Look up SRIR W-channel at nearest azimuth  → srir_w  [room acoustics]
  5. Build BRIR: brir_l/r = conv(srir_w, hrir_l/r)
  6. Spatialize: mono_source ★ BRIR → binaural source
  7. Sum all sources → mix → peak-normalise → save

All sources in a scene share the same HRTF subject + room/condition
(physically consistent) but have independent continuous (az, el) positions.
"""

import os
import sys
import json
import h5py
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import oaconvolve, resample_poly

# ── Project root on path so binaural_engine is importable ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from binaural_engine import load_sofa as _be_load_sofa, HRTFInterpolator

# ── Module-level paths ────────────────────────────────────────────────────────
HRTF_DIR     = os.path.join(_ROOT, 'hrtf')          # directory with p*.sofa files
SFX_DIR      = os.path.join(_ROOT, 'soud_effects')
SRIR_DIR     = os.path.join(_ROOT, 'sources', 'TAU_SRIR', 'TAU-SRIR_DB')
FSD50K_GT_DIR = os.path.join(_ROOT, 'sources', 'FSD50K', 'FSD50K.ground_truth')

# Kept for backward compat (single-subject fallback / custom HRTF use)
SOFA_PATH = os.path.join(HRTF_DIR, 'custom_mrs.sofa')

# ── Scene parameters ──────────────────────────────────────────────────────────
SCENE_DURATION  = 30.0    # seconds
HOP_SAMPLES     = 960     # 20 ms at 48 kHz
MAX_SLOTS       = 3       # maximum simultaneous sources per frame

# ── Continuous position ranges ────────────────────────────────────────────────
AZ_RANGE = (-180.0, 180.0)   # SLED CW degrees (0=front, positive=right)
EL_RANGE = ( -45.0,  45.0)   # elevation degrees

# ── Event scheduling ──────────────────────────────────────────────────────────
MAX_SIMULTANEOUS  = 3
NUM_EVENTS_RANGE  = (3, 8)
MIN_EVENT_DUR     = 2.0    # seconds
MAX_EVENT_DUR     = 12.0   # seconds
FADE_DURATION     = 0.05   # seconds

# ── SRIR constants ────────────────────────────────────────────────────────────
_SRIR_NATIVE_FS = 24_000   # TAU-SRIR native sample rate

# ── SRIR room registry ────────────────────────────────────────────────────────
# circular=True: 360 uniformly-spaced azimuths (0–359°, 1° step, CCW)
# circular=False: trajectory positions with unknown azimuths → random selection
_SRIR_ROOMS = {
    'bomb_shelter': {'file': 'rirs_01_bomb_shelter.mat', 'circular': True},
    'gym':          {'file': 'rirs_02_gym.mat',          'circular': True},
    'pb132':        {'file': 'rirs_03_pb132.mat',        'circular': True},
    'pc226':        {'file': 'rirs_04_pc226.mat',        'circular': True},
    'tc352':        {'file': 'rirs_10_tc352.mat',        'circular': True},
    'sa203':        {'file': 'rirs_05_sa203.mat',        'circular': False},
    'sc203':        {'file': 'rirs_06_sc203.mat',        'circular': False},
    'se203':        {'file': 'rirs_08_se203.mat',        'circular': False},
    'tb103':        {'file': 'rirs_09_tb103.mat',        'circular': False},
}

# ── SRIR split — keep val/test rooms unseen during training ───────────────────
# Train uses 7 rooms; val/test use 2 held-out rooms (one circular, one non-circ)
SRIR_TRAIN_ROOMS = ['bomb_shelter', 'gym', 'pb132', 'pc226', 'sa203', 'sc203', 'se203']
SRIR_EVAL_ROOMS  = ['tb103', 'tc352']


# =============================================================================
# 1.  SOFA / HRTF utilities
# =============================================================================

def scan_sofa_paths(hrtf_dir=None):
    """Return sorted list of p*.sofa paths.

    hrtf_dir can be:
      - a directory  → glob for p*.sofa inside it
      - a single .sofa file path → return [that file]
      - None → use HRTF_DIR constant

    Falls back to custom_mrs.sofa if no p*.sofa files are found in the directory.
    """
    if hrtf_dir is None:
        hrtf_dir = HRTF_DIR

    # Single file given directly
    if hrtf_dir.endswith('.sofa') and os.path.isfile(hrtf_dir):
        return [hrtf_dir]

    import glob
    paths = sorted(glob.glob(os.path.join(hrtf_dir, 'p*.sofa')))
    if not paths:
        fallback = os.path.join(hrtf_dir, 'custom_mrs.sofa')
        if os.path.exists(fallback):
            paths = [fallback]
    return paths


def load_sofa(filepath, target_sr=None):
    """Load a SOFA file.  Kept for backward compatibility.

    Returns
    -------
    hrir_l, hrir_r : (M, N)
    azimuths       : (M,)  SOFA CCW degrees (0=front, 90=left)
    elevations     : (M,)  degrees
    sr             : int
    """
    hrirs, positions, sr = _be_load_sofa(filepath, target_sr=target_sr)
    return hrirs[:, 0, :], hrirs[:, 1, :], positions[:, 0], positions[:, 1], sr


def _load_sofa_h5py(sofa_path, target_sr=None):
    """Load SOFA via h5py (safe for concurrent multiprocess reads)."""
    with h5py.File(sofa_path, 'r') as f:
        hrirs = np.array(f['Data.IR'][:], dtype=np.float64)       # (M, 2, N)
        pos   = np.array(f['SourcePosition'][:], dtype=np.float64) # (M, 3)
        sr    = int(np.array(f['Data.SamplingRate']).flat[0])

    positions = pos[:, :2]   # drop radius column → (M, 2) az/el

    if target_sr is not None and target_sr != sr:
        resampled = []
        for m in range(hrirs.shape[0]):
            l_ch = resample_poly(hrirs[m, 0], target_sr, sr)
            r_ch = resample_poly(hrirs[m, 1], target_sr, sr)
            resampled.append(np.stack([l_ch, r_ch]))
        hrirs = np.array(resampled, dtype=np.float64)
        sr    = target_sr

    return hrirs, positions, sr


def build_hrtf_interpolator(sofa_path, target_sr=None, n_neighbors=3):
    """Load one SOFA file and return an HRTFInterpolator.

    Uses h5py for reading (safe for concurrent multiprocess access).
    Falls back to binaural_engine.load_sofa if h5py fails.
    """
    try:
        hrirs, positions, _ = _load_sofa_h5py(sofa_path, target_sr=target_sr)
    except Exception:
        hrirs, positions, _ = _be_load_sofa(sofa_path, target_sr=target_sr)
    return HRTFInterpolator(hrirs, positions, n_neighbors=n_neighbors)


def pick_and_build_interpolator(rng, sofa_paths, target_sr=None, n_neighbors=3):
    """Randomly select one SOFA file and build an HRTFInterpolator.

    Parameters
    ----------
    rng        : np.random.RandomState
    sofa_paths : list[str]  from scan_sofa_paths()
    target_sr  : int or None

    Returns
    -------
    interpolator : HRTFInterpolator
    subject_name : str   e.g. 'p0042'
    """
    sofa_path    = sofa_paths[rng.randint(0, len(sofa_paths))]
    subject_name = os.path.splitext(os.path.basename(sofa_path))[0]
    interpolator = build_hrtf_interpolator(sofa_path, target_sr=target_sr,
                                           n_neighbors=n_neighbors)
    return interpolator, subject_name


# =============================================================================
# 2.  SRIR loader (TAU-SRIR_DB, W-channel only)
# =============================================================================

def preload_srir_condition(rng, fs, srir_dir=None, rooms=None, room_name=None):
    """Select a random SRIR room/condition and load all-azimuth W-channels.

    Opens the HDF5 once, loads the W-channel (FOA ch 0, omnidirectional)
    for every azimuth position in the chosen condition, then resamples to *fs*.
    Call this once per scene; index into the result per source via get_srir_w().

    Parameters
    ----------
    rng       : np.random.RandomState
    fs        : int   Target sample rate
    srir_dir  : str or None  Override for SRIR_DIR
    rooms     : list[str] or None
        Subset of room names to sample from.  Use SRIR_TRAIN_ROOMS for train
        and SRIR_EVAL_ROOMS for val/test.  None = all rooms.
    room_name : str or None  Force a specific room (overrides rooms)

    Returns
    -------
    srir_w_all : np.ndarray  (n_az, N_fs)  float32
    circular   : bool  True if room has uniform 1°-step coverage
    room_meta  : dict  {'room', 'cond_row', 'cond_col'}
    """
    if srir_dir is None:
        srir_dir = SRIR_DIR
    if room_name is None:
        room_pool = rooms if rooms is not None else list(_SRIR_ROOMS.keys())
        room_name = room_pool[rng.randint(0, len(room_pool))]

    info     = _SRIR_ROOMS[room_name]
    mat_path = os.path.join(srir_dir, info['file'])

    with h5py.File(mat_path, 'r') as f:
        foa_ds   = f['rirs/foa']
        n_rt60, n_dist = foa_ds.shape
        cond_row = rng.randint(0, n_rt60)
        cond_col = rng.randint(0, n_dist)
        ref  = foa_ds[cond_row, cond_col]
        data = f[ref]                                  # (n_az, 4, N_native)
        w_native = np.array(data[:, 0, :], dtype=np.float64)   # (n_az, N_native)

    # Resample all azimuths in one call (vectorised along time axis)
    w_fs = resample_poly(w_native, fs, _SRIR_NATIVE_FS, axis=1).astype(np.float32)

    return w_fs, info['circular'], {
        'room':     room_name,
        'cond_row': int(cond_row),
        'cond_col': int(cond_col),
    }


def get_srir_w(srir_w_all, circular, az_sled_deg, rng):
    """Return the W-channel row matching the given source azimuth.

    Parameters
    ----------
    srir_w_all  : (n_az, N_fs)  float32  from preload_srir_condition()
    circular    : bool
    az_sled_deg : float  Source azimuth, SLED CW degrees
    rng         : np.random.RandomState  Used only for non-circular rooms

    Returns
    -------
    srir_w : (N_fs,)  float32
    az_idx : int      Index into srir_w_all that was selected
    """
    n_az = srir_w_all.shape[0]
    if circular:
        # TAU-SRIR circular rooms: index k = k degrees CCW from front
        az_ccw = (-az_sled_deg) % 360.0   # SLED CW → CCW in [0, 360)
        az_idx = int(round(az_ccw)) % n_az
    else:
        # Non-circular rooms: trajectory positions with unmapped azimuths
        az_idx = rng.randint(0, n_az)
    return srir_w_all[az_idx], az_idx


# =============================================================================
# 3.  SFX loader
# =============================================================================

def scan_sfx_paths(sfx_dir, allowed_fnames=None):
    """Scan sfx_dir and return {key: path} without loading audio.

    Supports flat layout (sfx_dir/{clip}.wav) and
    class layout (sfx_dir/{class_label}/{clip}.wav).

    Parameters
    ----------
    allowed_fnames : set[str] or None
        If given, only include clips whose filename stem (without extension)
        is in this set.  Used to enforce train/val/test FSD50K splits.
    """
    supported = ('.mp3', '.wav')
    paths = {}
    for entry in sorted(os.scandir(sfx_dir), key=lambda e: e.name):
        if entry.is_dir(follow_symlinks=True):
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.name.lower().endswith(supported) and sub.is_file(follow_symlinks=True):
                    if allowed_fnames is not None:
                        stem = os.path.splitext(sub.name)[0]
                        if stem not in allowed_fnames:
                            continue
                    paths[f"{entry.name}/{sub.name}"] = sub.path
        elif entry.is_file(follow_symlinks=True) and entry.name.lower().endswith(supported):
            if allowed_fnames is None or os.path.splitext(entry.name)[0] in allowed_fnames:
                paths[entry.name] = entry.path
    return paths


def read_fsd50k_split_fnames(gt_dir=None, dataset_split='train'):
    """Return the set of FSD50K clip fname stems for the requested split.

    Splits
    ------
    'train'  : dev.csv rows where split == 'train'   (~36,796 clips)
    'val'    : dev.csv rows where split == 'val'     (~4,170 clips)
    'test'   : eval.csv (all rows)                   (~10,231 clips)

    Parameters
    ----------
    gt_dir        : str or None  Path to FSD50K.ground_truth/ (defaults to FSD50K_GT_DIR)
    dataset_split : str  'train', 'val', or 'test'

    Returns
    -------
    set[str]  — filename stems (e.g. {'104008', '100005', ...})
    """
    import csv
    if gt_dir is None:
        gt_dir = FSD50K_GT_DIR

    if dataset_split in ('train', 'val'):
        csv_path = os.path.join(gt_dir, 'dev.csv')
        fnames = set()
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if row['split'] == dataset_split:
                    fnames.add(str(row['fname']))
    elif dataset_split == 'test':
        csv_path = os.path.join(gt_dir, 'eval.csv')
        fnames = set()
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                fnames.add(str(row['fname']))
    else:
        raise ValueError(f"dataset_split must be 'train', 'val', or 'test', got {dataset_split!r}")

    return fnames


def scan_sfx_split_paths(sfx_dir=None, gt_dir=None, dataset_split='train'):
    """Scan sfx_dir and return only clips belonging to the given FSD50K split.

    Parameters
    ----------
    sfx_dir       : str or None  SFX root directory (defaults to SFX_DIR)
    gt_dir        : str or None  FSD50K.ground_truth/ directory
    dataset_split : str  'train', 'val', or 'test'

    Returns
    -------
    dict  {key: path}
    """
    if sfx_dir is None:
        sfx_dir = SFX_DIR
    allowed = read_fsd50k_split_fnames(gt_dir, dataset_split)
    return scan_sfx_paths(sfx_dir, allowed_fnames=allowed)


def load_sfx_from_paths(sfx_paths, target_fs, max_files=None, seed=None):
    """Load audio from a pre-scanned {key: path} dict."""
    entries = list(sfx_paths.items())
    if max_files is not None and max_files < len(entries):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(entries), max_files, replace=False)
        entries = [entries[i] for i in sorted(idx)]
    print(f"  [SFX] Loading {len(entries)} / {len(sfx_paths)} clips …")
    sfx = {}
    for key, path in entries:
        try:
            audio, _ = librosa.load(path, sr=target_fs, mono=True)
            sfx[key] = audio
        except Exception as e:
            print(f"  [SFX] Warning: could not load {key}: {e}")
    return sfx


def load_sfx(sfx_dir, target_fs, max_files=None, seed=None):
    """Scan and load all SFX clips from sfx_dir."""
    paths = scan_sfx_paths(sfx_dir)
    if not paths:
        return {}
    return load_sfx_from_paths(paths, target_fs, max_files=max_files, seed=seed)


# =============================================================================
# 4.  Class-map builder
# =============================================================================

def build_class_map(sfx_dict):
    """Assign a consecutive integer class-id to each SFX key.

    For class layout (key='Dog/12345.wav') all clips of the same class
    share one id.  For flat layout (key='crash.mp3') the stem is the class.
    """
    def _label(key):
        parts = key.split('/', 1)
        return parts[0] if len(parts) == 2 else os.path.splitext(parts[0])[0]

    unique_labels = sorted(set(_label(k) for k in sfx_dict.keys()))
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    return {key: label2id[_label(key)] for key in sfx_dict.keys()}


# =============================================================================
# 5.  Event scheduling — continuous position sampling
# =============================================================================

def schedule_events(sfx, n_samples, rng, fs):
    """Schedule non-overlapping events (up to MAX_SIMULTANEOUS at once).

    Positions are sampled uniformly from continuous ranges:
      azimuth   ∈ AZ_RANGE = (−180°, 180°)   SLED CW (0=front, +=right)
      elevation ∈ EL_RANGE = (−45°,  +45°)

    Returns
    -------
    list of dict with keys:
        file, start_sample, end_sample, start_time, end_time,
        azimuth [SLED CW deg], elevation [deg], gain, audio_segment
    Sorted by start_sample (onset order).
    """
    sfx_names = list(sfx.keys())
    duration  = n_samples / fs
    activity  = np.zeros(n_samples, dtype=np.int8)
    events    = []
    n_events  = rng.randint(NUM_EVENTS_RANGE[0], NUM_EVENTS_RANGE[1] + 1)

    for _attempt in range(n_events * 20):
        if len(events) >= n_events:
            break

        dur    = rng.uniform(MIN_EVENT_DUR, MAX_EVENT_DUR)
        t_max  = duration - dur
        if t_max <= 0:
            continue
        t_start = rng.uniform(0, t_max)
        t_end   = t_start + dur
        s_start = int(t_start * fs)
        s_end   = min(int(t_end * fs), n_samples)

        if activity[s_start:s_end].max() >= MAX_SIMULTANEOUS:
            continue

        fname = rng.choice(sfx_names)
        src   = sfx[fname]
        n_seg = s_end - s_start

        if len(src) < n_seg:
            reps = int(np.ceil(n_seg / max(len(src), 1)))
            seg  = np.tile(src, reps)[:n_seg].copy()
        else:
            offset = rng.randint(0, max(1, len(src) - n_seg))
            seg    = src[offset: offset + n_seg].copy()

        # Fade in/out to reduce clicks
        fade = max(1, min(int(FADE_DURATION * fs), n_seg // 4))
        seg[:fade]  *= np.linspace(0.0, 1.0, fade)
        seg[-fade:] *= np.linspace(1.0, 0.0, fade)

        # Normalise
        peak = np.max(np.abs(seg))
        if peak > 1e-8:
            seg /= peak

        # Continuous random position (SLED CW convention)
        az_deg = float(rng.uniform(*AZ_RANGE))
        el_deg = float(rng.uniform(*EL_RANGE))
        gain   = float(rng.uniform(0.3, 0.8))

        events.append({
            'file'         : fname,
            'start_sample' : s_start,
            'end_sample'   : s_end,
            'start_time'   : round(float(t_start), 4),
            'end_time'     : round(float(t_end),   4),
            'azimuth'      : round(az_deg, 4),   # SLED CW degrees
            'elevation'    : round(el_deg, 4),   # degrees
            'gain'         : round(gain,   4),
            'audio_segment': seg,
        })
        activity[s_start:s_end] += 1

    events.sort(key=lambda e: e['start_sample'])
    return events


# =============================================================================
# 6.  Binaural mixer — HRTF interpolation + SRIR room acoustics
# =============================================================================

def mix_binaural(events, interpolator, srir_w_all, srir_circular, n_samples, rng):
    """Spatialize events using interpolated HRTF + SRIR room acoustics.

    For each event:
      1. Interpolate HRTF at the exact continuous (az, el) via IDW
         → hrir_l, hrir_r  (encodes the precise head-shadow / ITD / ILD)
      2. Look up pre-loaded SRIR W-channel at nearest azimuth
         → srir_w  (omnidirectional; provides room reverb without direction bias)
      3. Build BRIR = conv(srir_w, hrir)  — direction from HRTF, reverb from SRIR
      4. Spatialize via overlap-add convolution: oaconvolve(seg, BRIR)[:n_seg]

    No separate noise is added; the SRIR reverb tail inherently carries
    the ambient noise of the measured room.

    Returns
    -------
    mix_L, mix_R : (n_samples,) float64, peak-normalised to ±0.9
    """
    mix_L = np.zeros(n_samples, dtype=np.float64)
    mix_R = np.zeros(n_samples, dtype=np.float64)

    for ev in events:
        az_sled = ev['azimuth']      # SLED CW degrees
        el      = ev['elevation']    # degrees
        gain    = ev['gain']
        seg     = ev['audio_segment'].astype(np.float64)
        s_start = ev['start_sample']
        s_end   = ev['end_sample']
        n_seg   = s_end - s_start

        # ── 1. Interpolate HRTF at exact continuous position ────────────────
        # binaural_engine convention: az=0 front, az=90 LEFT (CCW, SOFA)
        az_sofa = -az_sled           # SLED CW → SOFA CCW
        hrir_l, hrir_r = interpolator.interpolate(az_sofa, el)  # float64 (N_hrir,)

        # ── 2. SRIR W-channel for room acoustics ────────────────────────────
        srir_w, _ = get_srir_w(srir_w_all, srir_circular, az_sled, rng)
        srir_w    = srir_w.astype(np.float64)

        # ── 3. Build BRIR = conv(SRIR_W, HRIR) ─────────────────────────────
        # SRIR encodes room reflections; HRIR encodes head/ear transfer.
        # Convolving them gives the full Binaural Room Impulse Response.
        brir_l = np.convolve(srir_w, hrir_l)   # (N_srir + N_hrir − 1,)
        brir_r = np.convolve(srir_w, hrir_r)

        # ── 4. Spatialize via overlap-add ───────────────────────────────────
        sig_l = oaconvolve(seg, brir_l)[:n_seg]
        sig_r = oaconvolve(seg, brir_r)[:n_seg]

        mix_L[s_start:s_end] += gain * sig_l
        mix_R[s_start:s_end] += gain * sig_r

    # Peak normalise
    peak = max(np.max(np.abs(mix_L)), np.max(np.abs(mix_R)), 1e-8)
    mix_L *= 0.9 / peak
    mix_R *= 0.9 / peak

    return mix_L, mix_R


# =============================================================================
# 7.  Dense annotation computation
# =============================================================================

def _az_el_to_unit_vector(az_sled_deg, el_deg):
    """SLED (az, el) → unit-vector (x=fwd, y=right, z=up).

    SLED convention: az=0 front, az=+90° right (CW).
      x = cos(el) × cos(az)   [forward component]
      y = cos(el) × sin(az)   [rightward component, + = right]
      z = sin(el)             [upward component]
    """
    az = np.deg2rad(az_sled_deg)
    el = np.deg2rad(el_deg)
    return (
        float(np.cos(el) * np.cos(az)),
        float(np.cos(el) * np.sin(az)),
        float(np.sin(el)),
    )


def compute_dense_annotations(events, n_samples, class_map, fs):
    """Produce per-frame ground-truth annotation arrays.

    Frame t covers samples [t × HOP, (t+1) × HOP).
    Slots are filled in onset order; up to MAX_SLOTS sources per frame.

    Returns
    -------
    cls_arr  : (T, MAX_SLOTS)     int16   class_id, −1 = inactive
    doa_arr  : (T, MAX_SLOTS, 3)  float16  unit vector (x, y, z)
    loud_arr : (T, MAX_SLOTS)     float16  dBFS of dry mono per frame
    mask_arr : (T, MAX_SLOTS)     bool     active-slot flag
    """
    T = n_samples // HOP_SAMPLES

    cls_arr  = np.full((T, MAX_SLOTS), -1,    dtype=np.int16)
    doa_arr  = np.zeros((T, MAX_SLOTS, 3),    dtype=np.float16)
    loud_arr = np.full((T, MAX_SLOTS), -80.0, dtype=np.float16)
    mask_arr = np.zeros((T, MAX_SLOTS),       dtype=bool)

    eps = 1e-8

    for t in range(T):
        frame_start = t * HOP_SAMPLES
        frame_end   = frame_start + HOP_SAMPLES
        slot        = 0

        for ev in events:
            if slot >= MAX_SLOTS:
                break
            # Event is active in this frame if any overlap exists
            if ev['start_sample'] < frame_end and ev['end_sample'] > frame_start:
                seg_start = max(0, frame_start - ev['start_sample'])
                seg_end   = min(len(ev['audio_segment']),
                                frame_end   - ev['start_sample'])
                seg_frame = ev['audio_segment'][seg_start:seg_end]

                rms  = np.sqrt(np.mean((ev['gain'] * seg_frame) ** 2) + eps)
                loud = 20.0 * np.log10(rms)

                x, y, z = _az_el_to_unit_vector(ev['azimuth'], ev['elevation'])
                cls_id  = class_map.get(ev['file'], 0)

                cls_arr [t, slot]    = cls_id
                doa_arr [t, slot, :] = [x, y, z]
                loud_arr[t, slot]    = np.float16(loud)
                mask_arr[t, slot]    = True
                slot += 1

    return cls_arr, doa_arr, loud_arr, mask_arr


# =============================================================================
# 8.  Scene synthesizer
# =============================================================================

def synthesize_scene(name, sofa_paths, sfx, class_map, output_dir, fs,
                     srir_dir=None, srir_rooms=None, seed=None):
    """Synthesize one binaural scene and save all annotation files.

    A random HRTF subject is selected per scene from sofa_paths, and a
    random SRIR room + condition is selected from srir_rooms.

    Parameters
    ----------
    name       : str        Scene name (used as filename stem)
    sofa_paths : list[str]  SOFA file paths to sample from (e.g. all p*.sofa)
    sfx        : dict       {key: np.ndarray (mono float32)}
    class_map  : dict       {sfx_key: class_id}
    output_dir : str        Directory for output files
    fs         : int        Sample rate
    srir_dir   : str|None   Path to TAU-SRIR_DB (defaults to SRIR_DIR)
    srir_rooms : list[str]|None
        Room names to sample from.  Pass SRIR_TRAIN_ROOMS for train splits
        and SRIR_EVAL_ROOMS for val/test.  None = all rooms.
    seed       : int|None   RNG seed for reproducibility

    Saves
    -----
    {name}.wav          stereo float32 WAV
    {name}.json         scene metadata + event list
    {name}_cls.npy      [T, MAX_SLOTS]     int16   class ids
    {name}_doa.npy      [T, MAX_SLOTS, 3]  float16 DOA unit vectors
    {name}_loud.npy     [T, MAX_SLOTS]     float16 loudness dBFS
    {name}_mask.npy     [T, MAX_SLOTS]     bool    active-slot flag
    """
    rng       = np.random.RandomState(seed)
    n_samples = int(SCENE_DURATION * fs)

    # ── Random HRTF subject for this scene ───────────────────────────────────
    interpolator, hrtf_subject = pick_and_build_interpolator(
        rng, sofa_paths, target_sr=fs
    )

    # ── Random SRIR room + condition for this scene ───────────────────────────
    srir_w_all, srir_circular, room_meta = preload_srir_condition(
        rng, fs, srir_dir=srir_dir, rooms=srir_rooms
    )

    events = schedule_events(sfx, n_samples, rng, fs)
    mix_L, mix_R = mix_binaural(
        events, interpolator, srir_w_all, srir_circular, n_samples, rng
    )
    cls_arr, doa_arr, loud_arr, mask_arr = compute_dense_annotations(
        events, n_samples, class_map, fs
    )

    os.makedirs(output_dir, exist_ok=True)

    # WAV
    stereo   = np.stack([mix_L, mix_R], axis=1).astype(np.float32)
    wav_path = os.path.join(output_dir, f'{name}.wav')
    sf.write(wav_path, stereo, int(fs))

    # JSON metadata
    gt_events = [
        {k: v for k, v in ev.items() if k != 'audio_segment'}
        for ev in events
    ]
    meta = {
        'scene_name'   : name,
        'duration_sec' : SCENE_DURATION,
        'sample_rate'  : int(fs),
        'hop_samples'  : HOP_SAMPLES,
        'n_frames'     : int(n_samples // HOP_SAMPLES),
        'max_slots'    : MAX_SLOTS,
        'audio_file'   : f'{name}.wav',
        'num_events'   : len(events),
        'hrtf_subject' : hrtf_subject,
        'srir'         : room_meta,
        'events'       : sorted(gt_events, key=lambda e: e['start_time']),
    }
    json_path = os.path.join(output_dir, f'{name}.json')
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    # Dense annotation arrays
    np.save(os.path.join(output_dir, f'{name}_cls.npy'),  cls_arr)
    np.save(os.path.join(output_dir, f'{name}_doa.npy'),  doa_arr)
    np.save(os.path.join(output_dir, f'{name}_loud.npy'), loud_arr)
    np.save(os.path.join(output_dir, f'{name}_mask.npy'), mask_arr)

    return wav_path, json_path
