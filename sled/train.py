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
from torch.utils.tensorboard import SummaryWriter

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
# Loss computation
# =============================================================================

def _compute_single_layer_loss(pred: dict, gt: dict) -> torch.Tensor:
    """Compute matching loss for one decoder layer output.

    Vectorised implementation — eliminates the B×T GPU↔CPU sync loop:
      1. All cost matrices computed on GPU in a single batched operation.
      2. ONE GPU→CPU transfer for all [BT, S, S] cost matrices.
      3. scipy Hungarian runs in a tight CPU-only loop (no GPU ops inside).
      4. Matched indices uploaded to GPU once; all losses computed in batch.

    Losses
    ------
      - Class loss     : focal loss on matched (pred, GT-class) pairs only
      - DOA loss       : cosine distance on matched pairs
      - Presence loss  : BCE on all slots (1 = matched, 0 = unmatched/inactive)
      - Count loss     : L1 between Σ sigmoid(conf) and GT active count
    """
    class_logits = pred['class_logits']   # [B, T, S, C]
    doa_vec      = pred['doa_vec']        # [B, T, S, 3]
    confidence   = pred['confidence']     # [B, T, S]

    gt_cls  = gt['cls']    # [B, T, S]   int64
    gt_doa  = gt['doa']    # [B, T, S, 3]
    gt_mask = gt['mask']   # [B, T, S]   bool

    B, T, S, C = class_logits.shape
    T_gt = gt_cls.shape[1]
    if T != T_gt:
        T = min(T, T_gt)
        class_logits = class_logits[:, :T]
        doa_vec      = doa_vec[:, :T]
        confidence   = confidence[:, :T]
        gt_cls  = gt_cls[:, :T]
        gt_doa  = gt_doa[:, :T]
        gt_mask = gt_mask[:, :T]

    BT     = B * T
    device = class_logits.device

    # ── Flatten batch × time ──────────────────────────────────────────────────
    logits_flat = class_logits.reshape(BT, S, C)   # [BT, S, C]
    doa_flat    = doa_vec.reshape(BT, S, 3)        # [BT, S, 3]
    conf_flat   = confidence.reshape(BT, S)        # [BT, S]
    cls_gt_flat = gt_cls.reshape(BT, S)            # [BT, S]
    doa_gt_flat = gt_doa.reshape(BT, S, 3)         # [BT, S, 3]
    mask_flat   = gt_mask.reshape(BT, S)           # [BT, S] bool

    # ── Build all [BT, S, S] cost matrices on GPU in one pass ─────────────────
    # cls_cost ∈ [-1, 0]  (negative probability)
    # doa_cost ∈ [0, 2]   (1 − cosine similarity)
    # Scale doa by 0.5 so both terms have comparable magnitude ∈ [0, 1].
    with torch.no_grad():
        prob = logits_flat.softmax(-1)                            # [BT, S, C]

        # cls_cost[bt, i, j] = -prob[bt, i, cls_gt[bt, j]]
        # Clamp to [0, C-1]: inactive slots carry class=n_classes (=C),
        # which is out of range for gather. The clamped cost is a dummy —
        # inactive columns are excluded via active_cols = np.where(mask_np[bt]).
        cls_gt_exp = cls_gt_flat.clamp(0, C - 1).unsqueeze(1).expand(BT, S, S)
        cls_cost   = -prob.gather(2, cls_gt_exp)                  # [BT, S, S]

        # doa_cost[bt, i, j] = 1 - cos_sim(pred_doa[bt,i], gt_doa[bt,j])
        pred_norm = F.normalize(doa_flat,    dim=-1)              # [BT, S, 3]
        gt_norm   = F.normalize(doa_gt_flat, dim=-1)              # [BT, S, 3]
        doa_cost  = 1.0 - torch.bmm(pred_norm, gt_norm.transpose(1, 2))  # [BT, S, S]

        # ONE GPU→CPU transfer for all cost matrices + mask
        cost_np = (cls_cost + 0.5 * doa_cost).cpu().numpy()   # [BT, S, S]
        mask_np = mask_flat.cpu().numpy()                      # [BT, S] bool

    # ── Hungarian matching — pure CPU, zero GPU ops inside ────────────────────
    # Use actual active-column indices (not :n_act slice) so that col_arr always
    # refers to valid GT slots even when active slots are not contiguous.
    bt_arr  = np.empty(BT * S, dtype=np.int64)
    row_arr = np.empty(BT * S, dtype=np.int64)
    col_arr = np.empty(BT * S, dtype=np.int64)
    n_matched = 0

    for bt in range(BT):
        active_cols = np.where(mask_np[bt])[0]   # actual active GT slot indices
        if len(active_cols) == 0:
            continue
        r, c = linear_sum_assignment(cost_np[bt][:, active_cols])
        c = active_cols[c]   # remap back to original GT slot positions
        m = len(r)
        bt_arr [n_matched:n_matched + m] = bt
        row_arr[n_matched:n_matched + m] = r
        col_arr[n_matched:n_matched + m] = c
        n_matched += m

    # ── Vectorised loss computation — ONE upload, batch GPU ops ───────────────
    conf_tgt = torch.zeros(BT, S, device=device)

    if n_matched > 0:
        bt_idx  = torch.from_numpy(bt_arr [:n_matched]).to(device)
        row_idx = torch.from_numpy(row_arr[:n_matched]).to(device)
        col_idx = torch.from_numpy(col_arr[:n_matched]).to(device)

        conf_tgt[bt_idx, row_idx] = 1.0

        matched_pred_cls = logits_flat[bt_idx, row_idx]    # [M, C]
        matched_gt_cls   = cls_gt_flat[bt_idx, col_idx]    # [M]
        matched_pred_doa = doa_flat   [bt_idx, row_idx]    # [M, 3]
        matched_gt_doa   = doa_gt_flat[bt_idx, col_idx]    # [M, 3]

        cls_loss = focal_loss(matched_pred_cls, matched_gt_cls)
        doa_loss = cosine_dist_loss(matched_pred_doa, matched_gt_doa)
    else:
        cls_loss = torch.tensor(0.0, device=device)
        doa_loss = torch.tensor(0.0, device=device)

    presence_loss = F.binary_cross_entropy_with_logits(conf_flat, conf_tgt)

    return cls_loss + doa_loss + presence_loss


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

    total_loss = torch.tensor(0.0,
                               device=layer_preds[0]['class_logits'].device)

    for i, (w, pred) in enumerate(zip(weights, layer_preds)):
        match_loss = _compute_single_layer_loss(pred, gt)
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

            # Active DN slots: class id in valid range [0, C)
            # Inactive slots have class = n_classes (= C), used as padding in DN
            pos_mask = pos_cls_bt < C   # [B, T, G*S_sl]

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
            dn_loss = _compute_single_layer_loss(dn_pred_pos, dn_gt)

            # Negative DN slots (second half of S_dn) should have confidence=0.
            # This is the contrastive part: large-noise queries → inactive.
            dn_neg_conf = dn_conf[:, :, n_pos:]   # [B, T, n_neg]
            neg_conf_loss = F.binary_cross_entropy_with_logits(
                dn_neg_conf, torch.zeros_like(dn_neg_conf)
            )
            total_loss = total_loss + w * 0.5 * (dn_loss + neg_conf_loss)

    return total_loss


# =============================================================================
# Training / validation loops
# =============================================================================

def train_one_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer,
                    device: str, epoch: int,
                    writer: SummaryWriter, global_step: int) -> tuple[float, int]:
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
        global_step += 1
        writer.add_scalar('Loss/train_step', loss.item(), global_step)

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg = total_loss / n_batches
            print(f'  Epoch {epoch} [{batch_idx+1}/{len(loader)}]  '
                  f'loss={avg:.4f}  ({elapsed:.1f}s)')

    return total_loss / max(n_batches, 1), global_step


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
    parser.add_argument('--dataset-root',   default='./data_custom_hrtf')
    parser.add_argument('--sofa-path',      default='./hrtf/custom_mrs.sofa')
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch-size',     type=int,   default=64)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--workers',        type=int,   default=8)
    parser.add_argument('--device',         default='cuda' if torch.cuda.is_available()
                                                     else 'cpu')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--log-dir',        default='./runs',
                        help='TensorBoard log directory')
    parser.add_argument('--resume',         type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--window-frames',  type=int,   default=48,
                        help='Frames per training window (256 × 20ms = 5.12s)')
    parser.add_argument('--d-model',           type=int,   default=256)
    parser.add_argument('--n-classes',         type=int,   default=209)
    parser.add_argument('--min-loudness-db',   type=float, default=-60.0,
                        help='Slots quieter than this (dBFS) are treated as '
                             'inactive even if annotated as present')
    parser.add_argument('--no-hrtf-corr',  action='store_true', default=False,
                        help='Ablation: disable HRTF cross-corr heatmap')
    parser.add_argument('--no-ild',        action='store_true', default=False,
                        help='Ablation: disable ILD channel')
    parser.add_argument('--no-ipd',        action='store_true', default=False,
                        help='Ablation: disable IPD channels (sin/cos)')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f'[TB]   TensorBoard log dir: {os.path.abspath(args.log_dir)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    sofa_path = os.path.abspath(args.sofa_path)
    use_hrtf_corr = not args.no_hrtf_corr
    use_ild       = not args.no_ild
    use_ipd       = not args.no_ipd
    in_channels   = 2 + int(use_ild) + 2 * int(use_ipd)
    print(f'[MODEL] Building SLEDv3 (d={args.d_model}, classes={args.n_classes}, '
          f'in_ch={in_channels}, ild={use_ild}, ipd={use_ipd}, hrtf_corr={use_hrtf_corr})')
    model = SLEDv3(
        sofa_path      = sofa_path,
        d_model        = args.d_model,
        n_classes      = args.n_classes,
        use_hrtf_corr  = use_hrtf_corr,
        use_ild        = use_ild,
        use_ipd        = use_ipd,
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
        dataset_root    = dataset_root,
        split           = 'train',
        batch_size      = args.batch_size,
        window_frames   = args.window_frames,
        augment_scs     = True,
        num_workers     = args.workers,
        pin_memory      = use_pin,
        min_loudness_db = args.min_loudness_db,
    )
    val_loader = build_dataloader(
        dataset_root    = dataset_root,
        split           = 'val',
        batch_size      = args.batch_size,
        window_frames   = args.window_frames,
        augment_scs     = False,
        num_workers     = args.workers,
        pin_memory      = use_pin,
        min_loudness_db = args.min_loudness_db,
    )
    print(f'       Train: {len(train_loader.dataset)} scenes | '
          f'Val: {len(val_loader.dataset)} scenes')

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    global_step   = (start_epoch - 1) * len(train_loader)

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
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer, global_step
        )
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t_start
        lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:4d}/{args.epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={lr:.2e}  '
              f'({elapsed:.0f}s)')

        writer.add_scalars('Loss/epoch',
                           {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('LR', lr, epoch)

        # Save checkpoint every 10 epochs and when best
        if epoch % 10 == 0 or val_loss < best_val_loss:
            ckpt = {
                'epoch'         : epoch,
                'model'         : model.state_dict(),
                'optimizer'     : optimizer.state_dict(),
                'scheduler'     : scheduler.state_dict(),
                'val_loss'      : val_loss,
                'use_hrtf_corr' : use_hrtf_corr,
                'use_ild'       : use_ild,
                'use_ipd'       : use_ipd,
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

    writer.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
