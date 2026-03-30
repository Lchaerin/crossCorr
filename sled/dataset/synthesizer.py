#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Binaural Scene Synthesizer
=====================================
Synthesizes binaural audio scenes for SLED training.
Adapted from generate_audio.py with dense per-frame annotations.

Output per scene:
  {name}.wav           — stereo float32 WAV
  {name}.json          — scene metadata
  {name}_cls.npy       — [T, 5]    int16   class ids (-1 = inactive)
  {name}_doa.npy       — [T, 5, 3] float16 unit-vector (x,y,z)
  {name}_loud.npy      — [T, 5]    float16 loudness dB
  {name}_mask.npy      — [T, 5]    bool    active slot flag
"""

import os
import json
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

# ── Module-level paths ────────────────────────────────────────────────────────
SOFA_PATH      = os.path.join(os.path.dirname(__file__), '..', '..', 'hrtf', 'custom_mrs.sofa')
SFX_DIR        = os.path.join(os.path.dirname(__file__), '..', '..', 'soud_effects')

# ── Scene parameters ──────────────────────────────────────────────────────────
SCENE_DURATION  = 30.0    # seconds
HOP_SAMPLES     = 960     # 20 ms at 48 kHz
MAX_SLOTS       = 3       # maximum simultaneous sources per frame

# ── Event scheduling parameters ──────────────────────────────────────────────
MAX_SIMULTANEOUS  = 3
NUM_EVENTS_RANGE  = (3, 8)
MIN_EVENT_DUR     = 2.0   # seconds
MAX_EVENT_DUR     = 12.0  # seconds
FADE_DURATION     = 0.05  # seconds


# =============================================================================
# 1.  SOFA loader
# =============================================================================

def load_sofa(filepath):
    """Load HRIR data from a SOFA file.

    Returns
    -------
    hrir_l     : np.ndarray  (M, N)
    hrir_r     : np.ndarray  (M, N)
    azimuths   : np.ndarray  (M,)   degrees, SOFA convention (0=front, 90=left CCW)
    elevations : np.ndarray  (M,)   degrees
    fs_hrtf    : float               sampling rate
    """
    try:
        import netCDF4
        ds = netCDF4.Dataset(filepath, 'r')
        ir_data    = np.array(ds.variables['Data.IR'][:])         # (M, 2, N)
        source_pos = np.array(ds.variables['SourcePosition'][:])  # (M, 3)
        fs_hrtf    = float(ds.variables['Data.SamplingRate'][:].flat[0])
        ds.close()
    except Exception:
        import h5py
        with h5py.File(filepath, 'r') as f:
            ir_data    = np.array(f['Data.IR'])
            source_pos = np.array(f['SourcePosition'])
            fs_hrtf    = float(np.array(f['Data.SamplingRate']).flat[0])

    hrir_l     = ir_data[:, 0, :]   # (M, N)
    hrir_r     = ir_data[:, 1, :]
    azimuths   = source_pos[:, 0]   # degrees
    elevations = source_pos[:, 1]
    return hrir_l, hrir_r, azimuths, elevations, fs_hrtf


# =============================================================================
# 2.  SFX loader
# =============================================================================

def scan_sfx_paths(sfx_dir):
    """Scan sfx_dir and return {key: path} without loading any audio.

    Supports two layouts:
      Flat:  sfx_dir/{clip}.wav
      Class: sfx_dir/{class_label}/{clip}.wav  (FSD50K symlink layout)
    """
    supported = ('.mp3', '.wav')
    paths = {}
    for entry in sorted(os.scandir(sfx_dir), key=lambda e: e.name):
        if entry.is_dir(follow_symlinks=True):
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.name.lower().endswith(supported) and sub.is_file(follow_symlinks=True):
                    paths[f"{entry.name}/{sub.name}"] = sub.path
        elif entry.is_file(follow_symlinks=True) and entry.name.lower().endswith(supported):
            paths[entry.name] = entry.path
    return paths


def load_sfx_from_paths(sfx_paths, target_fs, max_files=None, seed=None):
    """Load audio from a pre-scanned {key: path} dict.

    Parameters
    ----------
    sfx_paths  : dict  {key: path}  from scan_sfx_paths()
    target_fs  : int   target sample rate
    max_files  : int or None  if set, randomly sample this many clips
    seed       : int or None  RNG seed for the sampling

    Returns
    -------
    dict  {key: np.ndarray (mono float32)}
    """
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
    """Scan sfx_dir and load audio clips.

    Supports two layouts:
      Flat:  sfx_dir/{clip}.wav
      Class: sfx_dir/{class_label}/{clip}.wav  (FSD50K symlink layout)

    Parameters
    ----------
    max_files : int or None  if set, randomly sample this many clips
    seed      : int or None  RNG seed for sampling

    Returns
    -------
    dict  {key: np.ndarray (mono float32)}
    """
    paths = scan_sfx_paths(sfx_dir)
    if not paths:
        return {}
    return load_sfx_from_paths(paths, target_fs, max_files=max_files, seed=seed)


# =============================================================================
# 3.  Class-map builder
# =============================================================================

def build_class_map(sfx_dict):
    """Assign a consecutive integer class-id to each sound-effect key.

    For flat layout  (key = 'crash.mp3') the class label is the stem.
    For class layout (key = 'Dog/12345.wav') the class label is the
    directory part ('Dog'), so all clips of the same class share one id.

    Returns
    -------
    dict  {sfx_key: class_id}   (class_id starts at 0)
    """
    # Collect unique class labels in sorted order
    def _label(key):
        parts = key.split('/', 1)
        return parts[0] if len(parts) == 2 else os.path.splitext(parts[0])[0]

    unique_labels = sorted(set(_label(k) for k in sfx_dict.keys()))
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    return {key: label2id[_label(key)] for key in sfx_dict.keys()}


# =============================================================================
# 4.  Event scheduling
# =============================================================================

def schedule_events(sfx, azimuths, elevations, n_samples, rng, fs):
    """Schedule non-overlapping (up to MAX_SIMULTANEOUS) audio events.

    Returns
    -------
    list of dict, each with keys:
        file, start_sample, end_sample, start_time, end_time,
        azimuth, elevation, az_idx, gain, audio_segment
    Events are sorted by start_sample (onset order).
    """
    sfx_names = list(sfx.keys())
    duration  = n_samples / fs

    activity = np.zeros(n_samples, dtype=np.int8)
    events   = []
    n_events = rng.randint(NUM_EVENTS_RANGE[0], NUM_EVENTS_RANGE[1] + 1)

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
        s_end   = int(t_end   * fs)
        s_end   = min(s_end, n_samples)

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

        # Fade in/out
        fade = max(1, min(int(FADE_DURATION * fs), n_seg // 4))
        ramp_in  = np.linspace(0.0, 1.0, fade)
        ramp_out = np.linspace(1.0, 0.0, fade)
        seg[:fade]  *= ramp_in
        seg[-fade:] *= ramp_out

        # Normalise
        peak = np.max(np.abs(seg))
        if peak > 1e-8:
            seg /= peak

        az_idx = rng.randint(0, len(azimuths))
        gain   = rng.uniform(0.3, 0.8)

        events.append({
            'file'         : fname,
            'start_sample' : s_start,
            'end_sample'   : s_end,
            'start_time'   : round(float(t_start), 4),
            'end_time'     : round(float(t_end),   4),
            'azimuth'      : float(azimuths[az_idx]),
            'elevation'    : float(elevations[az_idx]),
            'az_idx'       : int(az_idx),
            'gain'         : round(float(gain), 4),
            'audio_segment': seg,
        })
        activity[s_start:s_end] += 1

    # Sort by onset so slot-fill order is deterministic
    events.sort(key=lambda e: e['start_sample'])
    return events


# =============================================================================
# 5.  Binaural mixer
# =============================================================================

def mix_binaural(events, hrir_l, hrir_r, n_samples, rng):
    """Convolve each event with its HRTF and sum to a stereo mix.

    Adds background white noise at a random SNR between 15 and 35 dB.

    Returns
    -------
    mix_L, mix_R : np.ndarray  (n_samples,)  float64
    """
    mix_L = np.zeros(n_samples, dtype=np.float64)
    mix_R = np.zeros(n_samples, dtype=np.float64)

    for ev in events:
        seg     = ev['audio_segment']
        idx     = ev['az_idx']
        gain    = ev['gain']
        s_start = ev['start_sample']
        s_end   = ev['end_sample']
        n_seg   = s_end - s_start

        hl = hrir_l[idx, :]
        hr = hrir_r[idx, :]

        sig_l = fftconvolve(seg, hl, mode='full')[:n_seg]
        sig_r = fftconvolve(seg, hr, mode='full')[:n_seg]

        mix_L[s_start:s_end] += gain * sig_l
        mix_R[s_start:s_end] += gain * sig_r

    # Prevent clipping
    peak = max(np.max(np.abs(mix_L)), np.max(np.abs(mix_R)))
    if peak > 1e-8:
        mix_L *= 0.9 / peak
        mix_R *= 0.9 / peak

    # Background white noise
    snr_db    = rng.uniform(15.0, 35.0)
    sig_power = np.mean(mix_L ** 2 + mix_R ** 2) / 2.0
    if sig_power > 1e-12:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise_std   = np.sqrt(noise_power)
        mix_L += rng.randn(n_samples) * noise_std
        mix_R += rng.randn(n_samples) * noise_std

    return mix_L, mix_R


# =============================================================================
# 6.  Dense annotation computation
# =============================================================================

def _az_el_to_unit_vector(az_sofa_deg, el_deg):
    """Convert SOFA azimuth + elevation to SLED unit vector (x,y,z).

    SOFA: az=0 → front, az=90 → left  (counter-clockwise).
    SLED: az=0 → front, CW (negate SOFA az).

    Coordinate convention (right-handed, SLED):
        x = cos(el) * cos(az_sled)
        y = cos(el) * sin(az_sled)   ← positive = right
        z = sin(el)                  ← positive = up
    """
    az_sled_deg = (-az_sofa_deg) % 360.0
    az_rad = np.deg2rad(az_sled_deg)
    el_rad = np.deg2rad(el_deg)
    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)
    return float(x), float(y), float(z)


def compute_dense_annotations(events, n_samples, class_map, fs):
    """Produce per-frame ground-truth annotation arrays.

    Frame t covers samples [t*HOP_SAMPLES, (t+1)*HOP_SAMPLES).

    Returns
    -------
    cls_arr  : np.ndarray  [T, MAX_SLOTS]     int16   (-1 = inactive)
    doa_arr  : np.ndarray  [T, MAX_SLOTS, 3]  float16
    loud_arr : np.ndarray  [T, MAX_SLOTS]     float16  (dB)
    mask_arr : np.ndarray  [T, MAX_SLOTS]     bool
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
            # Event is active in this frame if there is any overlap
            if ev['start_sample'] < frame_end and ev['end_sample'] > frame_start:
                # Dry audio segment slice for this frame
                seg_start = max(0, frame_start - ev['start_sample'])
                seg_end   = min(len(ev['audio_segment']),
                                frame_end   - ev['start_sample'])
                seg_frame = ev['audio_segment'][seg_start:seg_end]

                rms   = np.sqrt(np.mean((ev['gain'] * seg_frame) ** 2) + eps)
                loud  = 20.0 * np.log10(rms)

                x, y, z = _az_el_to_unit_vector(ev['azimuth'], ev['elevation'])
                cls_id  = class_map.get(ev['file'], 0)

                cls_arr [t, slot]    = cls_id
                doa_arr [t, slot, :] = [x, y, z]
                loud_arr[t, slot]    = np.float16(loud)
                mask_arr[t, slot]    = True
                slot += 1

    return cls_arr, doa_arr, loud_arr, mask_arr


# =============================================================================
# 7.  Scene synthesizer
# =============================================================================

def synthesize_scene(name, hrir_l, hrir_r, azimuths, elevations,
                     sfx, class_map, output_dir, fs, seed=None):
    """Synthesize one binaural scene and save all annotation files.

    Saves
    -----
    {name}.wav, {name}.json,
    {name}_cls.npy, {name}_doa.npy, {name}_loud.npy, {name}_mask.npy
    """
    rng      = np.random.RandomState(seed)
    n_samples = int(SCENE_DURATION * fs)

    events   = schedule_events(sfx, azimuths, elevations, n_samples, rng, fs)
    mix_L, mix_R = mix_binaural(events, hrir_l, hrir_r, n_samples, rng)
    cls_arr, doa_arr, loud_arr, mask_arr = compute_dense_annotations(
        events, n_samples, class_map, fs
    )

    os.makedirs(output_dir, exist_ok=True)

    # WAV
    stereo = np.stack([mix_L, mix_R], axis=1).astype(np.float32)
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
        'events'       : sorted(gt_events, key=lambda e: e['start_time']),
    }
    json_path = os.path.join(output_dir, f'{name}.json')
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    # NumPy annotation arrays
    np.save(os.path.join(output_dir, f'{name}_cls.npy'),  cls_arr)
    np.save(os.path.join(output_dir, f'{name}_doa.npy'),  doa_arr)
    np.save(os.path.join(output_dir, f'{name}_loud.npy'), loud_arr)
    np.save(os.path.join(output_dir, f'{name}_mask.npy'), mask_arr)

    return wav_path, json_path
