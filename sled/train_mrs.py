#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — MRSLife/MRSSound 학습 스크립트
=========================================
두 가지 모드를 지원합니다:

  scratch  : MRSSound 데이터로 처음부터 학습 (27 MRS 클래스)
  finetune : 합성 데이터 사전학습 체크포인트를 MRSSound로 파인튜닝
             (클래스 헤드만 27 MRS 클래스로 교체, 나머지 가중치 로드)

Usage
-----
  # Scratch 학습
  python train_mrs.py --mode scratch \\
      --mrs-root ./MRSAudio/MRSLife/MRSSound \\
      --sofa-path ./hrtf/p0001.sofa \\
      --epochs 200 --batch-size 32

  # 파인튜닝
  python train_mrs.py --mode finetune \\
      --pretrained ./checkpoints/sled_best.pt \\
      --mrs-root ./MRSAudio/MRSLife/MRSSound \\
      --sofa-path ./hrtf/p0001.sofa \\
      --epochs 100 --batch-size 32 --lr 1e-4
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.dataset.mrs_dataset import build_mrs_dataloader, MRS_N_CLASSES
from sled.model.sled          import SLEDv3


# =============================================================================
# Loss utilities  (train.py와 동일)
# =============================================================================

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    N, C = logits.shape
    target_one_hot = torch.zeros_like(logits).scatter_(-1, targets.unsqueeze(-1), 1.0)
    p   = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target_one_hot, reduction='none')
    pt  = p * target_one_hot + (1 - p) * (1 - target_one_hot)
    at  = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)
    return (at * (1 - pt) ** gamma * bce).sum(-1).mean()


def cosine_dist_loss(pred_doa: torch.Tensor, gt_doa: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred_doa, gt_doa, dim=-1)).mean()


def _compute_single_layer_loss(pred: dict, gt: dict) -> torch.Tensor:
    class_logits = pred['class_logits']   # [B, T, S, C]
    doa_vec      = pred['doa_vec']        # [B, T, S, 3]
    confidence   = pred['confidence']     # [B, T, S]

    gt_cls  = gt['cls']    # [B, T, G]
    gt_doa  = gt['doa']    # [B, T, G, 3]
    gt_mask = gt['mask']   # [B, T, G]

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

    logits_flat = class_logits.reshape(BT, S, C)
    doa_flat    = doa_vec.reshape(BT, S, 3)
    conf_flat   = confidence.reshape(BT, S)
    cls_gt_flat = gt_cls.reshape(BT, -1)
    doa_gt_flat = gt_doa.reshape(BT, -1, 3)
    mask_flat   = gt_mask.reshape(BT, -1)

    G = cls_gt_flat.shape[1]   # GT 슬롯 수

    with torch.no_grad():
        prob = logits_flat.softmax(-1)   # [BT, S, C]

        cls_gt_exp = cls_gt_flat.clamp(0, C - 1).unsqueeze(1).expand(BT, S, G)
        cls_cost   = -prob.gather(2, cls_gt_exp)   # [BT, S, G]

        pred_norm = F.normalize(doa_flat,    dim=-1)   # [BT, S, 3]
        gt_norm   = F.normalize(doa_gt_flat, dim=-1)   # [BT, G, 3]
        doa_cost  = 1.0 - torch.bmm(pred_norm, gt_norm.transpose(1, 2))   # [BT, S, G]

        cost_np = (cls_cost + 0.5 * doa_cost).cpu().numpy()
        mask_np = mask_flat.cpu().numpy()

    bt_arr  = np.empty(BT * S, dtype=np.int64)
    row_arr = np.empty(BT * S, dtype=np.int64)
    col_arr = np.empty(BT * S, dtype=np.int64)
    n_matched = 0

    for bt in range(BT):
        active_cols = np.where(mask_np[bt])[0]
        if len(active_cols) == 0:
            continue
        r, c = linear_sum_assignment(cost_np[bt][:, active_cols])
        c = active_cols[c]
        m = len(r)
        bt_arr [n_matched:n_matched + m] = bt
        row_arr[n_matched:n_matched + m] = r
        col_arr[n_matched:n_matched + m] = c
        n_matched += m

    conf_tgt = torch.zeros(BT, S, device=device)

    if n_matched > 0:
        bt_idx  = torch.from_numpy(bt_arr [:n_matched]).to(device)
        row_idx = torch.from_numpy(row_arr[:n_matched]).to(device)
        col_idx = torch.from_numpy(col_arr[:n_matched]).to(device)

        conf_tgt[bt_idx, row_idx] = 1.0

        matched_pred_cls = logits_flat[bt_idx, row_idx]
        matched_gt_cls   = cls_gt_flat[bt_idx, col_idx]
        matched_pred_doa = doa_flat   [bt_idx, row_idx]
        matched_gt_doa   = doa_gt_flat[bt_idx, col_idx]

        cls_loss = focal_loss(matched_pred_cls, matched_gt_cls)
        doa_loss = cosine_dist_loss(matched_pred_doa, matched_gt_doa)
    else:
        cls_loss = torch.tensor(0.0, device=device)
        doa_loss = torch.tensor(0.0, device=device)

    presence_loss = F.binary_cross_entropy_with_logits(conf_flat, conf_tgt)
    return cls_loss + doa_loss + presence_loss


def compute_losses(layer_preds, gt, dn_preds=None, pos_tgt=None, neg_tgt=None):
    n_layers = len(layer_preds)
    base_weights = [0.2, 0.4, 0.6, 1.0]
    if n_layers <= len(base_weights):
        weights = base_weights[-n_layers:]
    else:
        weights = base_weights + [1.0] * (n_layers - len(base_weights))

    total_loss = torch.tensor(0.0, device=layer_preds[0]['class_logits'].device)

    for i, (w, pred) in enumerate(zip(weights, layer_preds)):
        total_loss = total_loss + w * _compute_single_layer_loss(pred, gt)

        if dn_preds is not None and pos_tgt is not None:
            dn_pred  = dn_preds[i]
            B, T_frames, S_dn, C = dn_pred['class_logits'].shape
            G = pos_tgt['cls'].shape[1] if pos_tgt['cls'].dim() == 3 else 1
            BT   = B * T_frames
            S_sl = S_dn // (G * 2)

            pos_cls_bt = pos_tgt['cls'].reshape(BT, G * S_sl)
            pos_doa_bt = pos_tgt['doa'].reshape(BT, G * S_sl, 3)
            pos_cls_bt = pos_cls_bt.reshape(B, T_frames, G * S_sl)
            pos_doa_bt = pos_doa_bt.reshape(B, T_frames, G * S_sl, 3)
            pos_mask   = pos_cls_bt < C

            n_pos         = G * S_sl
            dn_pos_logits = dn_pred['class_logits'].reshape(B, T_frames, S_dn, C)[:, :, :n_pos, :]
            dn_pos_doa    = dn_pred['doa_vec'      ].reshape(B, T_frames, S_dn, 3)[:, :, :n_pos, :]
            dn_pos_conf   = dn_pred['confidence'   ].reshape(B, T_frames, S_dn  )[:, :, :n_pos]

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

            dn_neg_conf  = dn_pred['confidence'].reshape(B, T_frames, S_dn)[:, :, n_pos:]
            neg_conf_loss = F.binary_cross_entropy_with_logits(
                dn_neg_conf, torch.zeros_like(dn_neg_conf)
            )
            total_loss = total_loss + w * 0.5 * (dn_loss + neg_conf_loss)

    return total_loss


# =============================================================================
# Train / Validate loops
# =============================================================================

def train_one_epoch(model, loader, optimizer, device, epoch, writer, global_step):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        audio = batch['audio'].to(device, non_blocking=True)
        gt = {
            'cls' : batch['cls'].to(device,  non_blocking=True),
            'doa' : batch['doa'].to(device,  non_blocking=True),
            'loud': batch['loud'].to(device, non_blocking=True),
            'mask': batch['mask'].to(device, non_blocking=True),
        }

        optimizer.zero_grad()
        result = model(audio, gt=gt)
        loss   = compute_losses(
            layer_preds = result['layer_preds'],
            gt          = gt,
            dn_preds    = result.get('dn_layer_preds'),
            pos_tgt     = result.get('pos_targets'),
            neg_tgt     = result.get('neg_targets'),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        n_batches   += 1
        global_step += 1
        writer.add_scalar('Loss/train_step', loss.item(), global_step)

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg = total_loss / n_batches
            print(f'  Epoch {epoch} [{batch_idx+1}/{len(loader)}]  '
                  f'loss={avg:.4f}  ({elapsed:.1f}s)')

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        audio = batch['audio'].to(device, non_blocking=True)
        gt = {
            'cls' : batch['cls'].to(device,  non_blocking=True),
            'doa' : batch['doa'].to(device,  non_blocking=True),
            'loud': batch['loud'].to(device, non_blocking=True),
            'mask': batch['mask'].to(device, non_blocking=True),
        }
        result = model(audio, gt=None)
        loss   = compute_losses(layer_preds=result['layer_preds'], gt=gt)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SLED v3 MRSSound 학습/파인튜닝',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 공통 인자
    parser.add_argument('--mode', choices=['scratch', 'finetune'], default='scratch',
                        help="'scratch': MRSSound으로 처음부터 학습  "
                             "'finetune': 사전학습 체크포인트에서 파인튜닝")
    parser.add_argument('--mrs-root',       default='./MRSAudio/MRSLife/MRSSound',
                        help='MRSSound 디렉터리 경로')
    parser.add_argument('--sofa-path',      default='./hrtf/p0001.sofa')
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch-size',     type=int,   default=32)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--workers',        type=int,   default=8)
    parser.add_argument('--device',         default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', default='./checkpoints_mrs')
    parser.add_argument('--log-dir',        default='./runs_mrs')
    parser.add_argument('--resume',         default=None,
                        help='중단된 MRS 학습 재개 체크포인트 (--mode와 무관)')
    parser.add_argument('--window-frames',  type=int,   default=256,
                        help='윈도우 프레임 수 (256 × 20ms = 5.12s)')
    parser.add_argument('--d-model',        type=int,   default=256)
    parser.add_argument('--min-loudness-db', type=float, default=-50.0)

    # scratch 전용
    parser.add_argument('--n-classes',      type=int,   default=MRS_N_CLASSES,
                        help='scratch 모드: 클래스 수 (기본 27 MRS 클래스)')

    # finetune 전용
    parser.add_argument('--pretrained',     default=None,
                        help='finetune 모드: 합성 데이터 사전학습 체크포인트 경로')
    parser.add_argument('--freeze-encoder', action='store_true', default=False,
                        help='finetune 모드: 인코더 가중치 동결 (head만 학습)')
    parser.add_argument('--use-fsd50k-cls', action='store_true', default=False,
                        help='finetune 모드: MRS 이벤트를 FSD50K 클래스 ID로 매핑 '
                             '(n_classes=209, 사전학습 클래스 헤드 재사용)')

    # 어블레이션 플래그 (사전학습 체크포인트와 일치시켜야 함)
    parser.add_argument('--no-hrtf-corr',   action='store_true', default=False)
    parser.add_argument('--no-ild',         action='store_true', default=False)
    parser.add_argument('--no-ipd',         action='store_true', default=False)

    args = parser.parse_args()

    # ── 파인튜닝 인자 검증 ────────────────────────────────────────────────────
    if args.mode == 'finetune' and args.pretrained is None:
        parser.error("--mode finetune 에는 --pretrained 가 필요합니다.")

    device = args.device
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f'[TB]   TensorBoard: {os.path.abspath(args.log_dir)}')

    # ── 모델 채널/클래스 설정 결정 ────────────────────────────────────────────
    use_hrtf_corr = not args.no_hrtf_corr
    use_ild       = not args.no_ild
    use_ipd       = not args.no_ipd

    if args.mode == 'finetune':
        # 사전학습 체크포인트에서 플래그 읽기 (체크포인트가 저장한 값 우선)
        ckpt_pre = torch.load(args.pretrained, map_location='cpu')
        use_hrtf_corr = ckpt_pre.get('use_hrtf_corr', use_hrtf_corr)
        use_ild       = ckpt_pre.get('use_ild',        use_ild)
        use_ipd       = ckpt_pre.get('use_ipd',        use_ipd)

        if args.use_fsd50k_cls:
            n_classes = 209   # 사전학습과 동일 → class head 재사용 가능
        else:
            n_classes = MRS_N_CLASSES   # 27 클래스로 교체
    else:
        n_classes = args.n_classes

    in_channels = 2 + int(use_ild) + 2 * int(use_ipd)
    print(f'[MODEL] SLEDv3 d={args.d_model}, classes={n_classes}, '
          f'in_ch={in_channels}, ild={use_ild}, ipd={use_ipd}, '
          f'hrtf_corr={use_hrtf_corr}, mode={args.mode}')

    model = SLEDv3(
        sofa_path     = os.path.abspath(args.sofa_path),
        d_model       = args.d_model,
        n_classes     = n_classes,
        use_hrtf_corr = use_hrtf_corr,
        use_ild       = use_ild,
        use_ipd       = use_ipd,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'        학습 파라미터: {n_params:,}')

    # ── 사전학습 가중치 로드 (finetune) ──────────────────────────────────────
    start_epoch = 1

    if args.mode == 'finetune' and args.resume is None:
        print(f'[FINETUNE] 사전학습 가중치 로드: {args.pretrained}')
        pretrained_state = ckpt_pre['model']

        if args.use_fsd50k_cls:
            # n_classes가 같으므로 완전 로드 가능
            missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        else:
            # 클래스 수가 다른 키 모두 제외:
            #   heads.class_head.*        — 분류 헤드
            #   denoising.class_embed.*   — DN 클래스 임베딩
            CLASS_KEYS = ('class_head', 'denoising.class_embed')
            filtered = {
                k: v for k, v in pretrained_state.items()
                if not any(ck in k for ck in CLASS_KEYS)
            }
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            unexpected_real = [u for u in unexpected if not any(ck in u for ck in CLASS_KEYS)]
            if unexpected_real:
                print(f'  WARNING: 로드되지 않은 예상 외 키: {unexpected_real[:5]}')

        class_head_missing = [m for m in missing if any(ck in m for ck in ('class_head', 'class_embed'))]
        other_missing      = [m for m in missing if not any(ck in m for ck in ('class_head', 'class_embed'))]
        print(f'  class 관련 키 초기화(신규): {len(class_head_missing)}개')
        if other_missing:
            print(f'  WARNING: class 관련 외 누락 키: {other_missing[:5]}')

        if args.freeze_encoder:
            # 인코더(전처리기 포함) 가중치 동결
            frozen = 0
            for name, param in model.named_parameters():
                if name.startswith('preprocessor') or name.startswith('encoder'):
                    param.requires_grad = False
                    frozen += param.numel()
            print(f'  인코더 동결: {frozen:,} 파라미터')

    # ── 옵티마이저 & 스케줄러 ────────────────────────────────────────────────
    # finetune 기본 LR은 scratch의 1/3 (커맨드라인에서 덮어쓸 수 있음)
    default_lr = 1e-4 if (args.mode == 'finetune' and args.lr == 3e-4) else args.lr
    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=default_lr, weight_decay=1e-3,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── MRS 학습 재개 ─────────────────────────────────────────────────────────
    if args.resume is not None:
        print(f'[RESUME] 체크포인트 로드: {args.resume}')
        ckpt_r = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt_r['model'])
        optimizer.load_state_dict(ckpt_r['optimizer'])
        scheduler.load_state_dict(ckpt_r['scheduler'])
        start_epoch = ckpt_r.get('epoch', 0) + 1
        print(f'         에포크 {start_epoch}부터 재개')

    # ── 데이터로더 ────────────────────────────────────────────────────────────
    mrs_root = os.path.abspath(args.mrs_root)
    print(f'[DATA]  MRSSound 경로: {mrs_root}')
    use_pin = str(device).startswith('cuda')
    use_fsd = args.use_fsd50k_cls and (args.mode == 'finetune')

    train_loader = build_mrs_dataloader(
        mrs_root       = mrs_root,
        split          = 'train',
        batch_size     = args.batch_size,
        window_frames  = args.window_frames,
        augment_scs    = True,
        num_workers    = args.workers,
        pin_memory     = use_pin,
        use_fsd50k_cls = use_fsd,
        min_loudness_db = args.min_loudness_db,
    )
    val_loader = build_mrs_dataloader(
        mrs_root       = mrs_root,
        split          = 'val',
        batch_size     = args.batch_size,
        window_frames  = args.window_frames,
        augment_scs    = False,
        num_workers    = args.workers,
        pin_memory     = use_pin,
        use_fsd50k_cls = use_fsd,
        min_loudness_db = args.min_loudness_db,
    )
    print(f'        Train: {len(train_loader.dataset)}개 세그먼트 | '
          f'Val: {len(val_loader.dataset)}개 세그먼트')

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    global_step   = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):

        # DN 그룹 수 커리큘럼 (scratch와 동일)
        # finetune은 짧게 돌리므로 처음부터 3그룹 유지
        if args.mode == 'scratch':
            model.denoising.n_dn_groups = 5 if epoch <= 30 else 3
        else:
            model.denoising.n_dn_groups = 3

        t_start = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer, global_step
        )
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t_start
        lr      = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:4d}/{args.epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={lr:.2e}  ({elapsed:.0f}s)')

        writer.add_scalars('Loss/epoch', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('LR', lr, epoch)

        # 체크포인트 저장
        if epoch % 10 == 0 or val_loss < best_val_loss:
            ckpt = {
                'epoch'         : epoch,
                'model'         : model.state_dict(),
                'optimizer'     : optimizer.state_dict(),
                'scheduler'     : scheduler.state_dict(),
                'val_loss'      : val_loss,
                'mode'          : args.mode,
                'n_classes'     : n_classes,
                'use_hrtf_corr' : use_hrtf_corr,
                'use_ild'       : use_ild,
                'use_ipd'       : use_ipd,
                'use_fsd50k_cls': use_fsd,
            }
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f'sled_mrs_{args.mode}_epoch_{epoch:04d}.pt'
            )
            torch.save(ckpt, ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint_dir, f'sled_mrs_{args.mode}_best.pt')
                torch.save(ckpt, best_path)
                print(f'  ** 최고 val_loss={best_val_loss:.4f} → {best_path}')

    writer.close()
    print(f'학습 완료 (mode={args.mode}).')


if __name__ == '__main__':
    main()
