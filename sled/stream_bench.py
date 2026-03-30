#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Streaming Latency & Metrics Benchmark
=================================================
여러 오디오 파일을 window 단위로 순차(B=1) 처리하며
실제 스트리밍 레이턴시 + 인식 성능을 동시에 측정한다.

  - 배치 처리 없음: 한 번에 window 1개씩
  - 인접 annotation npy가 있으면 Det-F1 / Cls-Acc / DOA-MAE 도 계산
  - CUDA Event로 GPU-accurate 타이밍

Usage
-----
    python -m sled.stream_bench \\
        --ckpt      checkpoints/sled_best.pt \\
        --audio     data/audio/test/scene_011000.wav \\
                    data/audio/test/scene_011001.wav \\
        [--sofa-path    ./hrtf/p0001.sofa] \\
        [--window-frames 48] \\
        [--conf-thresh   0.35] \\
        [--n-warmup      10] \\
        [--output-json   stream_results.jsonl]
"""

import argparse
import json
import os
import sys
import time
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.model.sled import SLEDv3
from sled.eval import evaluate_batch

HOP_SAMPLES = 960   # 20 ms at 48 kHz
FRAME_MS    = 20.0


def load_annotations(audio_path: Path):
    """scene_XXXXXX.wav 옆에 있는 annotation npy 파일들을 로드한다.
    audio/test/scene_011000.wav → annotations/test/scene_011000_{cls,doa,mask}.npy
    파일이 없으면 None 반환.
    """
    scene   = audio_path.stem                         # scene_011000
    # data/audio/test  →  data/annotations/test
    annot_dir = audio_path.parent.parent.parent / 'annotations' / audio_path.parent.name

    try:
        cls  = np.load(annot_dir / f'{scene}_cls.npy').astype(np.int64)     # [T, 3]
        doa  = np.load(annot_dir / f'{scene}_doa.npy').astype(np.float32)   # [T, 3, 3]
        loud = np.load(annot_dir / f'{scene}_loud.npy').astype(np.float32)  # [T, 3]
        mask = np.load(annot_dir / f'{scene}_mask.npy')                      # [T, 3] bool
        return cls, doa, loud, mask
    except FileNotFoundError as e:
        print(f'  [WARN] annotation not found: {e}')
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Streaming latency & metrics benchmark for SLED v3.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt',           required=True)
    parser.add_argument('--audio',          nargs='+', required=True,
                        help='One or more WAV files to process sequentially')
    parser.add_argument('--sofa-path',      default='./hrtf/p0001.sofa')
    parser.add_argument('--window-frames',  type=int,   default=48,
                        help='Number of 20ms frames per window')
    parser.add_argument('--conf-thresh',    type=float, default=0.35)
    parser.add_argument('--d-model',        type=int,   default=256)
    parser.add_argument('--n-classes',      type=int,   default=209)
    parser.add_argument('--n-warmup',       type=int,   default=10,
                        help='Number of warmup windows before measurement')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json',    default=None,
                        help='If set, append result dict to this JSON-lines file')
    args = parser.parse_args()

    device         = args.device
    window_samples = args.window_frames * HOP_SAMPLES
    window_audio_ms = args.window_frames * FRAME_MS

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f'[CKPT]  {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    use_hrtf_corr = ckpt.get('use_hrtf_corr', True)
    use_ild       = ckpt.get('use_ild',        True)
    use_ipd       = ckpt.get('use_ipd',        True)
    epoch         = ckpt.get('epoch', '?')
    print(f'        epoch={epoch}  ild={use_ild}  ipd={use_ipd}  hrtf={use_hrtf_corr}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SLEDv3(
        sofa_path     = os.path.abspath(args.sofa_path),
        d_model       = args.d_model,
        n_classes     = args.n_classes,
        use_hrtf_corr = use_hrtf_corr,
        use_ild       = use_ild,
        use_ipd       = use_ipd,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ── Loss function (same as training) ──────────────────────────────────────
    sys.path.insert(0, _HERE)
    from train import compute_losses

    use_cuda = device == 'cuda' or device.startswith('cuda')

    # ── Warmup ────────────────────────────────────────────────────────────────
    dummy = torch.zeros(1, 2, window_samples, device=device)
    with torch.no_grad():
        for _ in range(args.n_warmup):
            model(dummy)
    if use_cuda:
        torch.cuda.synchronize()
    print(f'[WARM]  {args.n_warmup} warmup windows done.')
    print(f'[FILES] {len(args.audio)} audio file(s), window={args.window_frames} frames ({window_audio_ms:.0f}ms)')

    # ── Accumulators ──────────────────────────────────────────────────────────
    window_times_ms = []
    total_windows   = 0

    acc_tp  = acc_fp  = acc_fn  = 0
    acc_cls_correct = acc_cls_total = 0
    acc_doa_errors  = []
    has_metrics     = False
    total_loss      = 0.0
    loss_windows    = 0

    # ── Per-file streaming loop ───────────────────────────────────────────────
    for audio_file in args.audio:
        audio_path = Path(audio_file)
        print(f'\n  [FILE] {audio_path.name}')

        # Load audio
        sr, data = wavfile.read(str(audio_path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)   # mono → stereo
        data = data[:, :2].T   # [2, N]
        if sr != 48_000:
            g = gcd(sr, 48_000)
            data = resample_poly(data, 48_000 // g, sr // g, axis=1).astype(np.float32)
        wav = torch.from_numpy(data)   # [2, N]

        n_windows = wav.shape[1] // window_samples
        if n_windows == 0:
            print('    [SKIP] too short')
            continue

        # Load GT annotations (optional)
        annot = load_annotations(audio_path)
        if annot is not None:
            cls_arr, doa_arr, loud_arr, mask_arr = annot
            has_metrics = True
            print(f'    GT loaded: {cls_arr.shape[0]} frames total')

        print(f'    {n_windows} windows × {window_audio_ms:.0f}ms')

        # Window-by-window inference
        with torch.no_grad():
            for i in range(n_windows):
                chunk = wav[:, i * window_samples:(i + 1) * window_samples]
                chunk = chunk.unsqueeze(0).to(device)   # [1, 2, window_samples]

                # ── Timed forward pass ────────────────────────────────────────
                if use_cuda:
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    result = model(chunk)
                    ev_e.record()
                    torch.cuda.synchronize()
                    elapsed = ev_s.elapsed_time(ev_e)
                else:
                    t0 = time.perf_counter()
                    result = model(chunk)
                    elapsed = (time.perf_counter() - t0) * 1000.0

                window_times_ms.append(elapsed)
                total_windows += 1

                # ── Metrics + Loss (if GT available) ─────────────────────────
                if annot is not None:
                    fs = i * args.window_frames
                    fe = (i + 1) * args.window_frames
                    if fe > cls_arr.shape[0]:
                        continue   # last partial window: skip

                    gt_cls  = torch.from_numpy(cls_arr[fs:fe]).unsqueeze(0).to(device)    # [1,T,3]
                    gt_doa  = torch.from_numpy(doa_arr[fs:fe]).unsqueeze(0).to(device)    # [1,T,3,3]
                    gt_loud = torch.from_numpy(loud_arr[fs:fe]).unsqueeze(0).to(device)   # [1,T,3]
                    gt_mask = torch.from_numpy(mask_arr[fs:fe]).unsqueeze(0).to(device)   # [1,T,3]

                    gt = {'cls': gt_cls, 'doa': gt_doa, 'loud': gt_loud, 'mask': gt_mask}
                    loss = compute_losses(result['layer_preds'], gt)
                    total_loss   += loss.item()
                    loss_windows += 1

                    pred = result['layer_preds'][-1]
                    acc  = evaluate_batch(
                        pred_cls   = pred['class_logits'],
                        pred_doa   = pred['doa_vec'],
                        pred_conf  = pred['confidence'],
                        gt_cls     = gt_cls,
                        gt_doa     = gt_doa,
                        gt_mask    = gt_mask,
                        conf_thresh = args.conf_thresh,
                    )
                    acc_tp  += acc['det_tp']
                    acc_fp  += acc['det_fp']
                    acc_fn  += acc['det_fn']
                    acc_cls_correct += acc['cls_correct']
                    acc_cls_total   += acc['cls_total']
                    acc_doa_errors.extend(acc['doa_errors'])

    if total_windows == 0:
        print('[ERROR] No windows processed.')
        sys.exit(1)

    # ── Latency statistics ────────────────────────────────────────────────────
    avg_window_ms  = statistics.mean(window_times_ms)
    med_window_ms  = statistics.median(window_times_ms)
    p99_window_ms  = sorted(window_times_ms)[int(len(window_times_ms) * 0.99)]
    max_window_ms  = max(window_times_ms)
    total_audio_ms = total_windows * window_audio_ms
    total_infer_ms = sum(window_times_ms)
    avg_frame_ms   = avg_window_ms / args.window_frames
    rtf            = total_infer_ms / total_audio_ms

    # ── Metrics ───────────────────────────────────────────────────────────────
    prec    = acc_tp / max(acc_tp + acc_fp, 1)
    rec     = acc_tp / max(acc_tp + acc_fn, 1)
    det_f1  = 2 * prec * rec / max(prec + rec, 1e-8)
    cls_acc = acc_cls_correct / max(acc_cls_total, 1)
    doa_mae = float(np.mean(acc_doa_errors)) if acc_doa_errors else float('nan')
    avg_loss = total_loss / max(loss_windows, 1) if has_metrics else float('nan')

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'  Files processed : {len(args.audio)}  ({total_windows} windows total)')
    print(f'  Window size     : {args.window_frames} frames = {window_audio_ms:.0f}ms audio')
    print(f'  ── avg/window   : {avg_window_ms:.3f} ms')
    print(f'  ── med/window   : {med_window_ms:.3f} ms')
    print(f'  ── p99/window   : {p99_window_ms:.3f} ms')
    print(f'  ── max/window   : {max_window_ms:.3f} ms')
    print(f'  ── avg ms/frame : {avg_frame_ms:.4f} ms')
    print(f'  ── RTF          : {rtf:.5f}  ({1/rtf:.0f}× faster than real-time)')
    print(f'  ── deadline slack: {window_audio_ms - avg_window_ms:.1f} ms per window')
    if has_metrics:
        print(f'  ── loss         : {avg_loss:.4f}')
        print(f'  ── det F1       : {det_f1:.4f}  (P={prec:.4f} R={rec:.4f})')
        print(f'  ── cls acc      : {cls_acc:.4f}  ({acc_cls_correct}/{acc_cls_total})')
        print(f'  ── DOA MAE      : {doa_mae:.2f}°  ({len(acc_doa_errors)} pairs)')
    print('=' * 60)

    result_dict = dict(
        ckpt            = args.ckpt,
        use_ild         = use_ild,
        use_ipd         = use_ipd,
        use_hrtf_corr   = use_hrtf_corr,
        n_files         = len(args.audio),
        window_frames   = args.window_frames,
        total_windows   = total_windows,
        avg_window_ms   = round(avg_window_ms,  3),
        med_window_ms   = round(med_window_ms,  3),
        p99_window_ms   = round(p99_window_ms,  3),
        max_window_ms   = round(max_window_ms,  3),
        avg_frame_ms    = round(avg_frame_ms,   4),
        rtf             = round(rtf,            6),
        loss            = round(avg_loss,       4) if has_metrics else None,
        det_f1          = round(det_f1,         4) if has_metrics else None,
        det_prec        = round(prec,           4) if has_metrics else None,
        det_rec         = round(rec,            4) if has_metrics else None,
        cls_acc         = round(cls_acc,        4) if has_metrics else None,
        doa_mae_deg     = round(doa_mae,        2) if has_metrics else None,
        n_matched       = len(acc_doa_errors)      if has_metrics else None,
    )

    if args.output_json:
        with open(args.output_json, 'a') as f:
            f.write(json.dumps(result_dict) + '\n')
        print(f'[OUT]   Appended to {args.output_json}')

    return result_dict


if __name__ == '__main__':
    main()
