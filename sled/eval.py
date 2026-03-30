#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Ablation Evaluation Script
======================================
Evaluates a checkpoint on the test split and reports:
  - val_loss      : Hungarian matching loss (same as training)
  - det_f1        : Frame-level source detection F1
  - cls_acc       : Class accuracy for Hungarian-matched (pred, GT) pairs
  - doa_mae       : Mean angular error (degrees) for matched pairs

Usage
-----
    python -m sled.eval \\
        --ckpt      checkpoints/some_run/sled_best.pt \\
        --dataset-root ./data \\
        [--sofa-path   ./hrtf/p0001.sofa] \\
        [--split       test] \\
        [--batch-size  8] \\
        [--conf-thresh 0.35] \\
        [--workers     4] \\
        [--device      cuda]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.dataset.torch_dataset import build_dataloader
from sled.model.sled import SLEDv3


# =============================================================================
# Metric helpers
# =============================================================================

def angular_error_deg(pred_vec: np.ndarray, gt_vec: np.ndarray) -> float:
    """Great-circle angle (degrees) between two unit vectors."""
    dot = float(np.clip((pred_vec * gt_vec).sum(), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def evaluate_batch(
    pred_cls:  torch.Tensor,   # [B, T, S, C] logits
    pred_doa:  torch.Tensor,   # [B, T, S, 3] unit vectors
    pred_conf: torch.Tensor,   # [B, T, S]    logits
    gt_cls:    torch.Tensor,   # [B, T, S_gt] int64
    gt_doa:    torch.Tensor,   # [B, T, S_gt, 3]
    gt_mask:   torch.Tensor,   # [B, T, S_gt] bool
    conf_thresh: float,
):
    """
    Returns dicts of per-frame accumulators:
        det_tp, det_fp, det_fn : detection counts (frame-level any-source)
        cls_correct, cls_total : matched classification counts
        doa_errors             : list of angular errors (degrees) for matched pairs
    """
    B, T, S_pred, C = pred_cls.shape
    T = min(T, gt_cls.shape[1])   # STFT may produce T+1 frames; align to GT
    S_gt = gt_cls.shape[2]

    pred_cls_np  = pred_cls.cpu().float().numpy()
    pred_doa_np  = F.normalize(pred_doa, dim=-1).cpu().float().numpy()
    pred_conf_np = torch.sigmoid(pred_conf).cpu().float().numpy()
    gt_cls_np    = gt_cls.cpu().numpy()
    gt_doa_np    = gt_doa.cpu().float().numpy()
    gt_mask_np   = gt_mask.cpu().numpy()

    det_tp = det_fp = det_fn = 0
    cls_correct = cls_total = 0
    doa_errors = []

    for b in range(B):
        for t in range(T):
            # Active GT slots
            active_gt = np.where(gt_mask_np[b, t])[0]
            # Active pred slots (above threshold)
            active_pred = np.where(pred_conf_np[b, t] >= conf_thresh)[0]

            n_gt   = len(active_gt)
            n_pred = len(active_pred)

            # Frame-level detection (any source present?)
            gt_present   = n_gt > 0
            pred_present = n_pred > 0
            if gt_present and pred_present:
                det_tp += 1
            elif gt_present and not pred_present:
                det_fn += 1
            elif not gt_present and pred_present:
                det_fp += 1
            # true negative: neither present → skip

            if n_gt == 0 or n_pred == 0:
                continue

            # Hungarian matching: cost = angular error between pred and GT DOA
            cost = np.zeros((n_pred, n_gt))
            for pi, p_idx in enumerate(active_pred):
                for gi, g_idx in enumerate(active_gt):
                    gt_v  = gt_doa_np[b, t, g_idx]
                    gt_v  = gt_v / (np.linalg.norm(gt_v) + 1e-8)
                    pr_v  = pred_doa_np[b, t, p_idx]
                    cost[pi, gi] = angular_error_deg(pr_v, gt_v)

            row_ind, col_ind = linear_sum_assignment(cost)

            for pi, gi in zip(row_ind, col_ind):
                p_slot = active_pred[pi]
                g_slot = active_gt[gi]

                # DOA error
                doa_errors.append(cost[pi, gi])

                # Class accuracy
                pred_class = int(pred_cls_np[b, t, p_slot].argmax())
                gt_class   = int(gt_cls_np[b, t, g_slot])
                if gt_class >= 0:   # -1 is inactive sentinel
                    cls_total   += 1
                    cls_correct += int(pred_class == gt_class)

    return dict(
        det_tp=det_tp, det_fp=det_fp, det_fn=det_fn,
        cls_correct=cls_correct, cls_total=cls_total,
        doa_errors=doa_errors,
    )


# =============================================================================
# Loss (same as training validate)
# =============================================================================

def compute_val_loss(model, loader, device):
    """Re-uses training loss for an apples-to-apples comparison."""
    # Import from train.py to avoid duplication
    sys.path.insert(0, _HERE)
    from train import compute_losses

    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device, non_blocking=True)
            gt = {k: batch[k].to(device, non_blocking=True)
                  for k in ('cls', 'doa', 'loud', 'mask')}
            result = model(audio, gt=None)
            loss = compute_losses(result['layer_preds'], gt)
            total += loss.item()
            n += 1
    return total / max(n, 1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a SLED v3 checkpoint on test data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt',           required=True)
    parser.add_argument('--dataset-root',   default='./data')
    parser.add_argument('--sofa-path',      default='./hrtf/p0001.sofa')
    parser.add_argument('--split',          default='test')
    parser.add_argument('--batch-size',     type=int,   default=8)
    parser.add_argument('--conf-thresh',    type=float, default=0.35)
    parser.add_argument('--workers',        type=int,   default=4)
    parser.add_argument('--d-model',        type=int,   default=256)
    parser.add_argument('--n-classes',      type=int,   default=209)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json',    default=None,
                        help='If set, append result dict to this JSON-lines file')
    args = parser.parse_args()

    device = args.device

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f'[CKPT]  {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    use_hrtf_corr = ckpt.get('use_hrtf_corr', True)
    use_ild       = ckpt.get('use_ild',        True)
    use_ipd       = ckpt.get('use_ipd',        True)
    epoch         = ckpt.get('epoch', '?')
    val_loss_ckpt = ckpt.get('val_loss', float('nan'))

    print(f'        epoch={epoch}  val_loss(train)={val_loss_ckpt:.4f}')
    print(f'        ild={use_ild}  ipd={use_ipd}  hrtf_corr={use_hrtf_corr}')

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

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = build_dataloader(
        dataset_root = args.dataset_root,
        split        = args.split,
        batch_size   = args.batch_size,
        num_workers  = args.workers,
        augment_scs  = False,
        shuffle      = False,
    )
    print(f'[DATA]  {args.split}: {len(loader.dataset)} scenes')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    acc_tp = acc_fp = acc_fn = 0
    acc_cls_correct = acc_cls_total = 0
    acc_doa_errors = []

    # Inference timing: GPU-accurate via CUDA events when available,
    # otherwise wall-clock time.  Measures only the model forward pass
    # (excludes data loading and metric computation).
    use_cuda_events = (device == 'cuda' or device.startswith('cuda'))
    total_infer_ms  = 0.0
    total_frames    = 0

    # Warmup: one silent forward pass so the first batch isn't penalised
    # by CUDA JIT / kernel launch overhead.
    _warmup = torch.zeros(1, 2, args.batch_size * 960, device=device)
    with torch.no_grad():
        model(_warmup)
    if use_cuda_events:
        torch.cuda.synchronize()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            audio = batch['audio'].to(device, non_blocking=True)
            gt = {k: batch[k].to(device, non_blocking=True)
                  for k in ('cls', 'doa', 'loud', 'mask')}

            # ── Timed forward pass ────────────────────────────────────────────
            if use_cuda_events:
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end   = torch.cuda.Event(enable_timing=True)
                ev_start.record()
                result = model(audio, gt=None)
                ev_end.record()
                torch.cuda.synchronize()
                batch_ms = ev_start.elapsed_time(ev_end)   # milliseconds
            else:
                t0 = time.perf_counter()
                result = model(audio, gt=None)
                batch_ms = (time.perf_counter() - t0) * 1000.0

            # frames = batch_size × window_frames
            n_frames_batch = audio.shape[0] * (audio.shape[2] // 960)
            total_infer_ms += batch_ms
            total_frames   += n_frames_batch
            # ─────────────────────────────────────────────────────────────────

            pred = result['layer_preds'][-1]   # last decoder layer

            acc = evaluate_batch(
                pred_cls  = pred['class_logits'],
                pred_doa  = pred['doa_vec'],
                pred_conf = pred['confidence'],
                gt_cls    = gt['cls'],
                gt_doa    = gt['doa'],
                gt_mask   = gt['mask'],
                conf_thresh = args.conf_thresh,
            )
            acc_tp  += acc['det_tp']
            acc_fp  += acc['det_fp']
            acc_fn  += acc['det_fn']
            acc_cls_correct += acc['cls_correct']
            acc_cls_total   += acc['cls_total']
            acc_doa_errors.extend(acc['doa_errors'])

            if (i + 1) % 20 == 0:
                print(f'  [{i+1}/{len(loader)}] batches processed')

    ms_per_frame = total_infer_ms / max(total_frames, 1)

    # ── Detection F1 ──────────────────────────────────────────────────────────
    prec = acc_tp / max(acc_tp + acc_fp, 1)
    rec  = acc_tp / max(acc_tp + acc_fn, 1)
    det_f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    # ── Class accuracy ────────────────────────────────────────────────────────
    cls_acc = acc_cls_correct / max(acc_cls_total, 1)

    # ── DOA MAE ───────────────────────────────────────────────────────────────
    doa_mae = float(np.mean(acc_doa_errors)) if acc_doa_errors else float('nan')

    # ── Compute val loss on same split ────────────────────────────────────────
    print('[LOSS]  Computing matching loss ...')
    test_loss = compute_val_loss(model, loader, device)

    # ── Report ────────────────────────────────────────────────────────────────
    result_dict = dict(
        ckpt          = args.ckpt,
        epoch         = epoch,
        use_ild       = use_ild,
        use_ipd       = use_ipd,
        use_hrtf_corr = use_hrtf_corr,
        split         = args.split,
        test_loss     = round(test_loss,    4),
        det_f1        = round(det_f1,       4),
        det_prec      = round(prec,         4),
        det_rec       = round(rec,          4),
        cls_acc       = round(cls_acc,      4),
        doa_mae_deg   = round(doa_mae,      2),
        n_matched     = len(acc_doa_errors),
        ms_per_frame  = round(ms_per_frame, 4),
        total_frames  = total_frames,
    )

    print()
    print('=' * 60)
    print(f'  Checkpoint : {os.path.basename(args.ckpt)}')
    print(f'  ILD={use_ild}  IPD={use_ipd}  HRTF={use_hrtf_corr}')
    print(f'  ── {args.split} loss  : {test_loss:.4f}')
    print(f'  ── det F1    : {det_f1:.4f}  (P={prec:.4f} R={rec:.4f})')
    print(f'  ── class acc : {cls_acc:.4f}  ({acc_cls_correct}/{acc_cls_total})')
    print(f'  ── DOA MAE   : {doa_mae:.2f}°  ({len(acc_doa_errors)} pairs)')
    print(f'  ── ms/frame  : {ms_per_frame:.4f} ms  ({total_frames} frames total)')
    print('=' * 60)

    if args.output_json:
        with open(args.output_json, 'a') as f:
            f.write(json.dumps(result_dict) + '\n')
        print(f'[OUT]   Appended to {args.output_json}')

    return result_dict


if __name__ == '__main__':
    main()
