#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Visualizer
=====================
Renders per-frame predicted vs ground-truth source directions as an MP4.

Usage
-----
    python -m sled.visualize \\
        --audio   path/to/scene.wav \\
        --ckpt    checkpoints/sled_best.pt \\
        [--gt-json    path/to/scene.json] \\
        [--class-map  data/meta/class_map.json] \\
        [--output     viz.mp4] \\
        [--fps        25] \\
        [--conf-thresh 0.35] \\
        [--window-frames 64] \\
        [--d-model    256] \\
        [--n-classes  210] \\
        [--sofa-path  hrtf/p0001.sofa] \\
        [--device     cuda]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.model.sled import SLEDv3

# ── Constants ──────────────────────────────────────────────────────────────────
HOP_SAMPLES = 960
FIG_W, FIG_H = 1400, 700   # pixels (figsize=(14,7), dpi=100)

# Up to 8 simultaneous GT events (tab10 palette)
_GT_COLORS   = plt.cm.tab10(np.linspace(0, 1, 10))[:8]
# One color per prediction slot
_SLOT_COLORS = ['#e74c3c', '#2ecc71', '#3498db']   # red, green, blue


# =============================================================================
# Utilities
# =============================================================================

def doa_to_az_el(doa: np.ndarray):
    """Unit vector [3] → (azimuth_deg, elevation_deg) in SLED convention."""
    x, y, z = doa
    el = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(y, x)) % 360.0)
    return az, el


def build_id_to_label(class_map_path: str | None) -> dict:
    """Build {class_id: label_str} from class_map.json.

    The empty/background class (max_id + 1) is automatically added as '(empty)'.
    """
    if not class_map_path or not os.path.exists(class_map_path):
        return {}
    with open(class_map_path) as f:
        cm = json.load(f)
    id_to_label = {}
    for key, cid in cm.items():
        parts = key.split('/', 1)
        label = parts[0] if len(parts) == 2 else os.path.splitext(parts[0])[0]
        if cid not in id_to_label:
            id_to_label[cid] = label
    # Empty/background class sits at max_id + 1
    if id_to_label:
        id_to_label[max(id_to_label) + 1] = '(empty)'
    return id_to_label


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, audio_np: np.ndarray, device: str,
                  window_frames: int = 64) -> tuple:
    """Slide model over full audio in non-overlapping windows.

    Returns
    -------
    cls_arr  : [T, S]    int   predicted class ids (argmax)
    doa_arr  : [T, S, 3] float predicted unit-vector DOAs
    conf_arr : [T, S]    float sigmoid confidence in [0, 1]
    """
    n_samples      = audio_np.shape[1]
    n_frames_total = n_samples // HOP_SAMPLES
    window_samples = window_frames * HOP_SAMPLES
    n_windows      = max(1, (n_frames_total + window_frames - 1) // window_frames)

    # Pad to exact multiple
    needed = n_windows * window_samples
    if audio_np.shape[1] < needed:
        audio_np = np.concatenate(
            [audio_np, np.zeros((2, needed - audio_np.shape[1]), dtype=np.float32)],
            axis=1,
        )

    all_cls, all_doa, all_conf = [], [], []
    model.eval()
    with torch.no_grad():
        for w in range(n_windows):
            s, e  = w * window_samples, (w + 1) * window_samples
            chunk = torch.from_numpy(audio_np[:, s:e]).unsqueeze(0).to(device)

            result = model(chunk, gt=None)
            pred   = result['layer_preds'][-1]   # last decoder layer

            all_cls.append(pred['class_logits'][0].argmax(-1).cpu().numpy())
            all_doa.append(pred['doa_vec'][0].cpu().numpy())
            all_conf.append(torch.sigmoid(pred['confidence'][0]).cpu().numpy())

    cls_arr  = np.concatenate(all_cls,  axis=0)[:n_frames_total]
    doa_arr  = np.concatenate(all_doa,  axis=0)[:n_frames_total]
    conf_arr = np.concatenate(all_conf, axis=0)[:n_frames_total]
    return cls_arr, doa_arr, conf_arr


# =============================================================================
# Ground-truth loader
# =============================================================================

def load_gt_per_frame(json_path: str, n_frames: int, sr: int = 48_000) -> list:
    """Parse scene JSON → list[list[dict]]  (per annotation frame)."""
    with open(json_path) as f:
        meta = json.load(f)

    gt_frames = []
    for t in range(n_frames):
        t_sec  = t * HOP_SAMPLES / sr
        active = []
        for ev in meta.get('events', []):
            if ev['start_time'] <= t_sec < ev['end_time']:
                az_sled = (-ev['azimuth']) % 360.0
                key     = ev['file']
                parts   = key.split('/', 1)
                label   = parts[0] if len(parts) == 2 else os.path.splitext(parts[0])[0]
                active.append({'azimuth': az_sled, 'elevation': ev['elevation'],
                               'label': label})
        gt_frames.append(active)

    return gt_frames


# =============================================================================
# Frame renderer
# =============================================================================

def render_frame(t: int, n_frames: int, sr: int,
                 cls_arr, doa_arr, conf_arr,
                 gt_frames, id_to_label: dict,
                 audio_wave_L: np.ndarray,
                 conf_thresh: float) -> np.ndarray:
    """Return one rendered frame as a [H, W, 3] uint8 RGB array."""
    fig = plt.figure(figsize=(14, 7), dpi=100, facecolor='#1a1a2e')
    gs  = GridSpec(2, 3, figure=fig,
                   width_ratios=[1.8, 0.05, 1.0], height_ratios=[5, 1],
                   hspace=0.08, wspace=0.25,
                   left=0.05, right=0.97, top=0.93, bottom=0.08)

    ax_polar = fig.add_subplot(gs[0, 0], projection='polar', facecolor='#0d1b2a')
    ax_info  = fig.add_subplot(gs[0, 2], facecolor='#0d1b2a')
    ax_wave  = fig.add_subplot(gs[1, :], facecolor='#0d1b2a')

    t_sec      = t * HOP_SAMPLES / sr
    total_sec  = n_frames * HOP_SAMPLES / sr
    fig.text(0.5, 0.97,
             f'SLED v3  ·  {t_sec:.2f} s  /  {total_sec:.1f} s',
             ha='center', va='top', color='white', fontsize=13, fontweight='bold')

    # ── Polar plot (top-down azimuth view) ────────────────────────────────────
    ax_polar.set_theta_zero_location('N')    # 0° = front = top
    ax_polar.set_theta_direction(-1)          # clockwise
    ax_polar.set_ylim(0, 1.05)
    ax_polar.set_yticks([0.259, 0.5, 0.707, 0.866, 1.0])
    ax_polar.set_yticklabels(['75°', '60°', '45°', '30°', '0°'],
                             fontsize=6.5, color='#888888')
    ax_polar.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_polar.set_xticklabels(['Front', '45°', 'Right', '135°',
                              'Back', '225°', 'Left', '315°'],
                             color='#cccccc', fontsize=7.5)
    ax_polar.tick_params(colors='#444444')
    ax_polar.spines['polar'].set_color('#334155')
    ax_polar.grid(color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
    ax_polar.set_title('Top-Down View  (radius = cos elevation)',
                       color='#888888', fontsize=8, pad=10)

    # GT events
    gt_events = gt_frames[t] if gt_frames is not None else []
    for i, ev in enumerate(gt_events):
        color  = _GT_COLORS[i % len(_GT_COLORS)]
        az_rad = np.radians(ev['azimuth'])
        r      = np.cos(np.radians(ev['elevation']))
        ax_polar.scatter(az_rad, r, s=200, c=[color], marker='o',
                         edgecolors='white', linewidths=1.2, zorder=5, alpha=0.9)
        ax_polar.text(az_rad, r + 0.06, ev['label'][:12],
                      color=color, fontsize=6.5, ha='center', va='bottom')

    # Predictions (always shown; dimmed when below conf_thresh)
    for s_idx in range(cls_arr.shape[1]):
        conf   = conf_arr[t, s_idx]
        az, el = doa_to_az_el(doa_arr[t, s_idx])
        az_rad = np.radians(az)
        r      = np.cos(np.radians(el))
        color  = _SLOT_COLORS[s_idx % len(_SLOT_COLORS)]
        cls_id = int(cls_arr[t, s_idx])
        label  = id_to_label.get(cls_id, f'cls{cls_id}')
        alpha  = 0.95 if conf >= conf_thresh else max(0.25, conf)
        ax_polar.scatter(az_rad, r, s=220, c=[color], marker='*',
                         edgecolors='white', linewidths=0.5, zorder=6, alpha=alpha)
        ax_polar.text(az_rad, r - 0.12,
                      f'{label[:12]}\n{conf:.2f}',
                      color=color, fontsize=6, ha='center', va='top', alpha=alpha)

    legend_elems = [
        mpatches.Patch(color='#aaaaaa', label='●  Ground truth'),
        mpatches.Patch(color='#aaaaaa', label='★  Prediction'),
    ]
    ax_polar.legend(handles=legend_elems, loc='lower right',
                    fontsize=7, facecolor='#1a1a2e', edgecolor='#334155',
                    labelcolor='#dddddd', framealpha=0.85)

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.axis('off')
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

    def _text(y, s, color='#dddddd', bold=False, size=8):
        ax_info.text(0.04, y, s, transform=ax_info.transAxes,
                     color=color, fontsize=size,
                     fontweight='bold' if bold else 'normal',
                     va='top', fontfamily='monospace')

    y = 0.97
    _text(y, 'PREDICTIONS', '#ffffff', bold=True, size=9); y -= 0.07
    for s_idx in range(cls_arr.shape[1]):
        conf   = conf_arr[t, s_idx]
        color  = _SLOT_COLORS[s_idx % len(_SLOT_COLORS)]
        az, el = doa_to_az_el(doa_arr[t, s_idx])
        cls_id = int(cls_arr[t, s_idx])
        label  = id_to_label.get(cls_id, f'cls{cls_id}')
        # Dim text colour when below threshold
        text_color = color if conf >= conf_thresh else '#777777'
        _text(y, f'Slot {s_idx+1}  {label[:16]}', text_color); y -= 0.055
        _text(y, f'  az={az:6.1f}°  el={el:+5.1f}°  p={conf:.2f}',
              text_color); y -= 0.055

    if gt_events:
        y -= 0.02
        _text(y, 'GROUND TRUTH', '#ffffff', bold=True, size=9); y -= 0.07
        for i, ev in enumerate(gt_events):
            color = _GT_COLORS[i % len(_GT_COLORS)]
            _text(y, f'• {ev["label"][:20]}', tuple(color[:3])); y -= 0.055
            _text(y, f'  az={ev["azimuth"]:6.1f}°  el={ev["elevation"]:+5.1f}°',
                  tuple(color[:3])); y -= 0.055

    # ── Waveform strip ────────────────────────────────────────────────────────
    # Show a 5-second window centred on current time
    half    = sr * 2           # ±2s
    cs      = t * HOP_SAMPLES
    s_start = max(0, cs - half)
    s_end   = min(len(audio_wave_L), s_start + sr * 4)
    s_start = max(0, s_end - sr * 4)

    t_axis = np.arange(s_start, s_end) / sr
    ax_wave.fill_between(t_axis, audio_wave_L[s_start:s_end], 0,
                         color='#3b82f6', alpha=0.65, linewidth=0)
    ax_wave.axvline(t_sec, color='#ef4444', linewidth=1.8, zorder=5)
    ax_wave.set_xlim(t_axis[0], t_axis[-1])
    peak = max(float(np.abs(audio_wave_L[s_start:s_end]).max()), 1e-3)
    ax_wave.set_ylim(-peak * 1.3, peak * 1.3)
    ax_wave.set_facecolor('#0d1b2a')
    ax_wave.tick_params(colors='#888888', labelsize=7)
    for sp in ax_wave.spines.values():
        sp.set_color('#334155')
    ax_wave.set_xlabel('Time (s)', color='#888888', fontsize=8)

    # ── Export ────────────────────────────────────────────────────────────────
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualise SLED v3 predictions as an MP4.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--audio',          required=True)
    parser.add_argument('--ckpt',           required=True)
    parser.add_argument('--gt-json',        default=None)
    parser.add_argument('--class-map',      default=None)
    parser.add_argument('--output',         default='viz.mp4')
    parser.add_argument('--fps',            type=int,   default=25)
    parser.add_argument('--conf-thresh',    type=float, default=0.35)
    parser.add_argument('--window-frames',  type=int,   default=64)
    parser.add_argument('--d-model',        type=int,   default=256)
    parser.add_argument('--n-classes',      type=int,   default=209)
    parser.add_argument('--sofa-path',      default='./hrtf/p0001.sofa')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── Model ─────────────────────────────────────────────────────────────────
    print('[MODEL] Building SLEDv3 ...')
    model = SLEDv3(
        sofa_path  = os.path.abspath(args.sofa_path),
        d_model    = args.d_model,
        n_classes  = args.n_classes,
    ).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    print(f'        epoch={ckpt.get("epoch","?")}  '
          f'val_loss={ckpt.get("val_loss", float("nan")):.4f}')

    # ── Audio ─────────────────────────────────────────────────────────────────
    print('[AUDIO] Loading ...')
    audio_np, sr = sf.read(args.audio, dtype='float32', always_2d=True)
    audio_np = audio_np.T                          # [2, N]
    n_frames = audio_np.shape[1] // HOP_SAMPLES
    print(f'        {audio_np.shape[1]/sr:.1f}s  sr={sr}  frames={n_frames}')

    # ── Inference ─────────────────────────────────────────────────────────────
    print('[INFER] Running ...')
    cls_arr, doa_arr, conf_arr = run_inference(
        model, audio_np.copy(), args.device, args.window_frames
    )

    # ── GT ────────────────────────────────────────────────────────────────────
    gt_frames = None
    if args.gt_json:
        print('[GT]    Loading ground truth ...')
        gt_frames = load_gt_per_frame(args.gt_json, n_frames, sr=sr)

    id_to_label = build_id_to_label(args.class_map)

    # ── Render → ffmpeg pipe ──────────────────────────────────────────────────
    anno_fps    = sr / HOP_SAMPLES          # 50
    frame_skip  = max(1, round(anno_fps / args.fps))
    frame_ids   = list(range(0, n_frames, frame_skip))
    audio_wave_L = audio_np[0]

    print(f'[RENDER] {len(frame_ids)} frames  @ {args.fps} fps  '
          f'(1 video frame = {frame_skip} annotation frames)')

    tmp_video = args.output.replace('.mp4', '_noaudio.mp4')
    ffmpeg_video = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{FIG_W}x{FIG_H}', '-pix_fmt', 'rgb24',
        '-r', str(args.fps), '-i', '-',
        '-c:v', 'mpeg4', '-q:v', '3',
        '-pix_fmt', 'yuv420p', tmp_video,
    ]
    proc = subprocess.Popen(ffmpeg_video, stdin=subprocess.PIPE)

    try:
        from tqdm import tqdm
        _iter = tqdm(frame_ids, desc='render', unit='fr')
    except ImportError:
        _iter = frame_ids

    try:
        for t in _iter:
            rgb = render_frame(
                t, n_frames, sr,
                cls_arr, doa_arr, conf_arr,
                gt_frames, id_to_label,
                audio_wave_L, args.conf_thresh,
            )
            proc.stdin.write(rgb.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    # ── Mux audio ─────────────────────────────────────────────────────────────
    print('[MUX]   Adding audio ...')
    subprocess.run([
        'ffmpeg', '-y',
        '-i', tmp_video,
        '-i', args.audio,
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
        '-shortest', args.output,
    ], check=True)
    os.remove(tmp_video)

    print(f'[DONE]  → {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
