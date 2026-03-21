#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Training Script
==========================
Trains SLEDv3 with:
  - Hungarian matching loss on matching queries
  - Contrastive denoising loss on DN queries
  - Cosine-annealing LR schedule
  - Curriculum phases for DN group count

Usage
-----
    python train.py [--dataset-root PATH] [--sofa-path PATH] [--epochs N]
                   [--batch-size N] [--lr F] [--workers N] [--device STR]
                   [--checkpoint-dir PATH] [--resume PATH] [--window-frames N]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Make sled package importable when run as a script from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.dataset.torch_dataset import build_dataloader
from sled.model.sled            import SLEDv3


# =============================================================================
# Loss utilities
# =============================================================================

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Sigmoid focal loss.

    Parameters
    ----------
    logits  : [N, C]  raw (un-sigmoided) class logits
    targets : [N]     int64 class indices in [0, C)

    Returns
    -------
    scalar mean loss
    """
    N, C = logits.shape
    # One-hot encode targets
    target_one_hot = torch.zeros_like(logits).scatter_(
        -1, targets.unsqueeze(-1), 1.0
    )   # [N, C]

    p   = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target_one_hot,
                                              reduction='none')   # [N, C]
    pt  = p * target_one_hot + (1 - p) * (1 - target_one_hot)
    at  = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)
    loss = at * (1 - pt) ** gamma * bce   # [N, C]
    return loss.sum(-1).mean()


def cosine_dist_loss(pred_doa: torch.Tensor,
                     gt_doa: torch.Tensor) -> torch.Tensor:
    """1 − cosine similarity, averaged over all elements.

    Parameters
    ----------
    pred_doa : [..., 3]  predicted unit vectors
    gt_doa   : [..., 3]  ground-truth unit vectors

    Returns
    -------
    scalar mean loss in [0, 2]
    """
    cos_sim = F.cosine_similarity(pred_doa, gt_doa, dim=-1)   # [...]
    return (1.0 - cos_sim).mean()


# =============================================================================
# Hungarian matching
# =============================================================================

def hungarian_match(pred_logits: torch.Tensor,
                    pred_doa: torch.Tensor,
                    gt_cls: torch.Tensor,
                    gt_doa: torch.Tensor,
                    gt_mask: torch.Tensor):
    """Per-sample Hungarian matching between predictions and GT.

    Matching cost = class cost + doa cost.

    Parameters
    ----------
    pred_logits : [N_pred, n_classes]
    pred_doa    : [N_pred, 3]
    gt_cls      : [N_gt]   int64
    gt_doa      : [N_gt, 3]
    gt_mask     : [N_gt]   bool  — only match against active GT entries

    Returns
    -------
    pred_indices : list[int]   matched prediction row indices
    gt_indices   : list[int]   matched GT row indices (within active set)
    active_idx   : LongTensor  indices of active GT entries in original [N_gt]
    """
    active_idx = gt_mask.nonzero(as_tuple=False).squeeze(1)
    if active_idx.numel() == 0:
        return [], [], active_idx

    gt_cls_a  = gt_cls[active_idx]    # [M]
    gt_doa_a  = gt_doa[active_idx]    # [M, 3]
    N, M      = pred_logits.shape[0], gt_cls_a.shape[0]

    with torch.no_grad():
        # Class cost: negative softmax probability for GT class
        prob      = pred_logits.softmax(-1)           # [N, C]
        cls_cost  = -prob[:, gt_cls_a]                # [N, M]

        # DOA cost: 1 − cos_sim between each pred and each GT
        p_norm = F.normalize(pred_doa, dim=-1)        # [N, 3]
        g_norm = F.normalize(gt_doa_a, dim=-1)        # [M, 3]
        doa_cost = 1.0 - p_norm @ g_norm.T            # [N, M]

        cost = (cls_cost + doa_cost).cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind.tolist(), col_ind.tolist(), active_idx


# =============================================================================
# Loss computation
# =============================================================================

def _compute_single_layer_loss(pred: dict, gt: dict,
                                empty_class_id: int) -> torch.Tensor:
    """Compute matching loss for one decoder layer output.

    Parameters
    ----------
    pred           : dict with class_logits, doa_vec, loudness, confidence
                     each [B, T, S, *]
    gt             : dict with cls, doa, loud, mask  [B, T, S, *]
    empty_class_id : index of the "empty" class

    Returns
    -------
    scalar loss
    """
    class_logits = pred['class_logits']   # [B, T, S, C]
    doa_vec      = pred['doa_vec']        # [B, T, S, 3]
    confidence   = pred['confidence']     # [B, T, S]

    gt_cls  = gt['cls']                   # [B, T, S]   int64
    gt_doa  = gt['doa']                   # [B, T, S, 3]
    gt_mask = gt['mask']                  # [B, T, S]   bool

    B, T, S, C = class_logits.shape
    # Align T: STFT may produce 1 extra frame vs annotations
    T_gt = gt_cls.shape[1]
    if T != T_gt:
        T = min(T, T_gt)
        class_logits = class_logits[:, :T]
        doa_vec      = doa_vec[:, :T]
        confidence   = confidence[:, :T]
        gt_cls  = gt_cls[:, :T]
        gt_doa  = gt_doa[:, :T]
        gt_mask = gt_mask[:, :T]

    total_cls_loss  = torch.tensor(0.0, device=class_logits.device)
    total_doa_loss  = torch.tensor(0.0, device=class_logits.device)
    total_conf_loss = torch.tensor(0.0, device=class_logits.device)
    count = 0

    for b in range(B):
        for t in range(T):
            logits_bt = class_logits[b, t]   # [S, C]
            doa_bt    = doa_vec[b, t]         # [S, 3]
            conf_bt   = confidence[b, t]      # [S]
            cls_gt_bt = gt_cls[b, t]          # [S]
            doa_gt_bt = gt_doa[b, t]          # [S, 3]
            mask_bt   = gt_mask[b, t]         # [S] bool

            if not mask_bt.any():
                # No active sources: all predictions should be empty class
                empty_tgt = torch.full((S,), empty_class_id,
                                       dtype=torch.long,
                                       device=logits_bt.device)
                total_cls_loss  = total_cls_loss  + focal_loss(logits_bt, empty_tgt)
                total_conf_loss = total_conf_loss + F.binary_cross_entropy_with_logits(
                    conf_bt,
                    torch.zeros_like(conf_bt),
                )
                count += 1
                continue

            # Hungarian match
            pred_idx, gt_idx, active_idx = hungarian_match(
                logits_bt, doa_bt, cls_gt_bt, doa_gt_bt, mask_bt
            )

            if not pred_idx:
                continue

            # Build target class tensor: empty for unmatched, GT class for matched
            tgt_cls = torch.full((S,), empty_class_id,
                                  dtype=torch.long, device=logits_bt.device)
            tgt_cls_list = active_idx[gt_idx]   # global active indices
            for pi, ci in zip(pred_idx, range(len(gt_idx))):
                tgt_cls[pi] = cls_gt_bt[active_idx[gt_idx[ci]]]

            total_cls_loss = total_cls_loss + focal_loss(logits_bt, tgt_cls)

            # DOA loss only on matched active slots
            if pred_idx:
                p_doa = doa_bt[pred_idx]                     # [M_matched, 3]
                g_doa = doa_gt_bt[active_idx[gt_idx]]        # [M_matched, 3]
                total_doa_loss = total_doa_loss + cosine_dist_loss(p_doa, g_doa)

            # Confidence: 1 for matched, 0 for unmatched
            conf_tgt = torch.zeros(S, device=conf_bt.device)
            for pi in pred_idx:
                conf_tgt[pi] = 1.0
            total_conf_loss = total_conf_loss + F.binary_cross_entropy_with_logits(
                conf_bt, conf_tgt
            )

            count += 1

    if count == 0:
        return torch.tensor(0.0, device=class_logits.device, requires_grad=True)

    return (total_cls_loss + total_doa_loss + total_conf_loss) / count


def compute_losses(layer_preds: list, gt: dict,
                   dn_preds: list | None = None,
                   pos_tgt: dict | None = None,
                   neg_tgt: dict | None = None) -> torch.Tensor:
    """Compute total training loss across all decoder layers.

    Layer weights: [0.2, 0.4, 0.6, 1.0] (last layer weighted most).

    Parameters
    ----------
    layer_preds : list of n_layers dicts (matching query predictions)
    gt          : ground-truth dict {cls, doa, loud, mask}
    dn_preds    : (optional) list of n_layers dicts (DN predictions)
    pos_tgt     : DN positive targets {cls [BT, G, S], doa [BT, G, S, 3]}
    neg_tgt     : DN negative targets (same structure)

    Returns
    -------
    total_loss : scalar tensor
    """
    n_layers = len(layer_preds)
    # Layer weights — pad to any n_layers
    base_weights = [0.2, 0.4, 0.6, 1.0]
    if n_layers <= len(base_weights):
        weights = base_weights[-n_layers:]
    else:
        weights = base_weights + [1.0] * (n_layers - len(base_weights))

    empty_class_id = layer_preds[0]['class_logits'].shape[-1] - 1

    total_loss = torch.tensor(0.0,
                               device=layer_preds[0]['class_logits'].device)

    for i, (w, pred) in enumerate(zip(weights, layer_preds)):
        match_loss = _compute_single_layer_loss(pred, gt, empty_class_id)
        total_loss = total_loss + w * match_loss

        # DN loss (positive pairs only — DN targets are aligned by construction)
        if dn_preds is not None and pos_tgt is not None:
            dn_pred = dn_preds[i]
            # dn_pred['class_logits']: [B, T, S_dn, C]
            # S_dn = G * 2 * S_slots
            B, T_frames, S_dn, C = dn_pred['class_logits'].shape
            G = pos_tgt['cls'].shape[1] if pos_tgt['cls'].dim() == 3 else 1

            # Reshape pos targets for loss computation
            # pos_tgt['cls']: [B*T, G, S_slots] → need to match S_dn layout
            BT    = B * T_frames
            S_sl  = S_dn // (G * 2)  # n_slots

            dn_cls_logits = dn_pred['class_logits'].reshape(B, T_frames, S_dn, C)
            dn_doa        = dn_pred['doa_vec'].reshape(B, T_frames, S_dn, 3)
            dn_conf       = dn_pred['confidence'].reshape(B, T_frames, S_dn)

            # Build GT for positive DN slots (first G*S_sl slots = positives)
            pos_cls_bt = pos_tgt['cls'].reshape(BT, G * S_sl)   # [BT, G*S_sl]
            pos_doa_bt = pos_tgt['doa'].reshape(BT, G * S_sl, 3)
            pos_cls_bt = pos_cls_bt.reshape(B, T_frames, G * S_sl)
            pos_doa_bt = pos_doa_bt.reshape(B, T_frames, G * S_sl, 3)

            pos_mask = pos_cls_bt >= 0   # [B, T, G*S_sl]

            # Only compute DN loss on positive slots (first half of S_dn)
            n_pos = G * S_sl
            dn_pos_logits = dn_cls_logits[:, :, :n_pos, :]
            dn_pos_doa    = dn_doa[:, :, :n_pos, :]
            dn_pos_conf   = dn_conf[:, :, :n_pos]

            dn_gt = {
                'cls' : pos_cls_bt,
                'doa' : pos_doa_bt,
                'loud': torch.zeros_like(pos_cls_bt, dtype=torch.float),
                'mask': pos_mask,
            }
            dn_pred_pos = {
                'class_logits': dn_pos_logits,
                'doa_vec'     : dn_pos_doa,
                'loudness'    : torch.zeros_like(dn_pos_conf),
                'confidence'  : dn_pos_conf,
            }
            dn_loss = _compute_single_layer_loss(dn_pred_pos, dn_gt,
                                                  empty_class_id)
            total_loss = total_loss + w * 0.5 * dn_loss   # half-weight for DN

    return total_loss


# =============================================================================
# Training / validation loops
# =============================================================================

def train_one_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer,
                    device: str, epoch: int) -> float:
    model.train()
    total_loss  = 0.0
    n_batches   = 0
    t0          = time.time()

    for batch_idx, batch in enumerate(loader):
        audio = batch['audio'].to(device, non_blocking=True)
        gt    = {
            'cls' : batch['cls'].to(device,  non_blocking=True),
            'doa' : batch['doa'].to(device,  non_blocking=True),
            'loud': batch['loud'].to(device, non_blocking=True),
            'mask': batch['mask'].to(device, non_blocking=True),
        }

        optimizer.zero_grad()
        result = model(audio, gt=gt)

        loss = compute_losses(
            layer_preds = result['layer_preds'],
            gt          = gt,
            dn_preds    = result.get('dn_layer_preds'),
            pos_tgt     = result.get('pos_targets'),
            neg_tgt     = result.get('neg_targets'),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg = total_loss / n_batches
            print(f'  Epoch {epoch} [{batch_idx+1}/{len(loader)}]  '
                  f'loss={avg:.4f}  ({elapsed:.1f}s)')

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        audio = batch['audio'].to(device, non_blocking=True)
        gt    = {
            'cls' : batch['cls'].to(device,  non_blocking=True),
            'doa' : batch['doa'].to(device,  non_blocking=True),
            'loud': batch['loud'].to(device, non_blocking=True),
            'mask': batch['mask'].to(device, non_blocking=True),
        }

        result = model(audio, gt=None)    # inference mode (no DN)
        loss = compute_losses(
            layer_preds = result['layer_preds'],
            gt          = gt,
        )
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train SLED v3.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset-root',   default='./data')
    parser.add_argument('--sofa-path',      default='./hrtf/p0001.sofa')
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch-size',     type=int,   default=8)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--workers',        type=int,   default=4)
    parser.add_argument('--device',         default='cuda' if torch.cuda.is_available()
                                                     else 'cpu')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--resume',         type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--window-frames',  type=int,   default=256,
                        help='Frames per training window (256 × 20ms = 5.12s)')
    parser.add_argument('--d-model',        type=int,   default=256)
    parser.add_argument('--n-classes',      type=int,   default=301)
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    sofa_path = os.path.abspath(args.sofa_path)
    print(f'[MODEL] Building SLEDv3 (d={args.d_model}, classes={args.n_classes})')
    model = SLEDv3(
        sofa_path  = sofa_path,
        d_model    = args.d_model,
        n_classes  = args.n_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'        Trainable parameters: {n_params:,}')

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume is not None:
        print(f'[RESUME] Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'         Resuming from epoch {start_epoch}')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    dataset_root = os.path.abspath(args.dataset_root)
    print(f'[DATA] Loading from {dataset_root}')
    use_pin = (device.type == torch.device('cuda').type if hasattr(device, 'type') else str(device).startswith('cuda'))
    train_loader = build_dataloader(
        dataset_root  = dataset_root,
        split         = 'train',
        batch_size    = args.batch_size,
        window_frames = args.window_frames,
        augment_scs   = True,
        num_workers   = args.workers,
        pin_memory    = use_pin,
    )
    val_loader = build_dataloader(
        dataset_root  = dataset_root,
        split         = 'val',
        batch_size    = args.batch_size,
        window_frames = args.window_frames,
        augment_scs   = False,
        num_workers   = args.workers,
        pin_memory    = use_pin,
    )
    print(f'       Train: {len(train_loader.dataset)} scenes | '
          f'Val: {len(val_loader.dataset)} scenes')

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):

        # Curriculum: adjust n_dn_groups
        # Phase 1 (1–30):  5 DN groups
        # Phase 2 (31–80): 3 DN groups
        # Phase 3 (81+):   3 DN groups
        if epoch <= 30:
            model.denoising.n_dn_groups = 5
        else:
            model.denoising.n_dn_groups = 3

        t_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t_start
        print(f'Epoch {epoch:4d}/{args.epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'({elapsed:.0f}s)')

        # Save checkpoint every 10 epochs and when best
        if epoch % 10 == 0 or val_loss < best_val_loss:
            ckpt = {
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss' : val_loss,
            }
            ckpt_path = os.path.join(
                args.checkpoint_dir, f'sled_epoch_{epoch:04d}.pt'
            )
            torch.save(ckpt, ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint_dir, 'sled_best.pt')
                torch.save(ckpt, best_path)
                print(f'  ** New best: val_loss={best_val_loss:.4f} → {best_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
