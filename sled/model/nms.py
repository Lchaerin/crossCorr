#!/home/rllab/anaconda3/bin/python
"""
SLED v3.2 — DOA-aware Non-Maximum Suppression
===============================================
Over-provisioned slots (12) → NMS → final detections (≤ max_sources).

NMS 조건: 두 슬롯의 DOA cosine similarity > cos_thresh AND 같은 predicted class
→ confidence가 낮은 쪽 제거.
"""

import torch
import torch.nn.functional as F


def doa_nms(
    class_logits: torch.Tensor,
    doa_vecs: torch.Tensor,
    confidences: torch.Tensor,
    cos_thresh: float = 0.9,
    conf_thresh: float = 0.5,
    max_sources: int = 3,
) -> torch.Tensor:
    """
    Parameters
    ----------
    class_logits : [S, C]   raw logits (pre-softmax)
    doa_vecs     : [S, 3]   unit vectors
    confidences  : [S]      after sigmoid
    cos_thresh   : cosine similarity threshold for suppression
    conf_thresh  : minimum confidence to consider a slot active
    max_sources  : maximum number of output detections

    Returns
    -------
    keep_idx : [K]  indices of kept slots, K ≤ max_sources
    """
    # Step 1: confidence threshold
    active_mask = confidences > conf_thresh
    active_idx  = active_mask.nonzero(as_tuple=True)[0]

    if len(active_idx) == 0:
        return active_idx

    # Step 2: sort by confidence descending
    order      = confidences[active_idx].argsort(descending=True)
    active_idx = active_idx[order]

    # Step 3: predicted class for each active slot
    pred_cls = class_logits[active_idx].argmax(dim=-1)   # [K_active]

    # Step 4: greedy NMS
    keep = []
    suppressed = torch.zeros(len(active_idx), dtype=torch.bool,
                             device=doa_vecs.device)

    for i in range(len(active_idx)):
        if suppressed[i]:
            continue

        idx_i = active_idx[i]
        keep.append(idx_i)

        if len(keep) >= max_sources:
            break

        # Suppress remaining slots that have same class AND similar DOA
        remaining = ~suppressed
        remaining[i] = False
        remaining_idx = remaining.nonzero(as_tuple=True)[0]

        if len(remaining_idx) == 0:
            continue

        cos_sim = F.cosine_similarity(
            doa_vecs[idx_i].unsqueeze(0),
            doa_vecs[active_idx[remaining_idx]],
            dim=-1
        )
        same_class = pred_cls[remaining_idx] == pred_cls[i]

        # Suppress: same class AND close DOA
        to_suppress = (cos_sim > cos_thresh) & same_class
        suppressed[remaining_idx[to_suppress]] = True

    if len(keep) == 0:
        return torch.tensor([], dtype=torch.long, device=doa_vecs.device)

    return torch.stack(keep)


def batch_doa_nms(
    class_logits: torch.Tensor,
    doa_vecs: torch.Tensor,
    confidences: torch.Tensor,
    cos_thresh: float = 0.9,
    conf_thresh: float = 0.5,
    max_sources: int = 3,
) -> list:
    """
    Batched NMS for [B, T, S, ...] shaped predictions.

    Parameters
    ----------
    class_logits : [B, T, S, C]
    doa_vecs     : [B, T, S, 3]
    confidences  : [B, T, S]      raw logits (sigmoid applied internally)

    Returns
    -------
    list of [B, T] tensors, each containing kept slot indices for that frame
    """
    B, T, S, C = class_logits.shape
    conf_prob = torch.sigmoid(confidences)   # logits → probability

    results = []
    for b in range(B):
        frame_results = []
        for t in range(T):
            keep = doa_nms(
                class_logits[b, t],   # [S, C]
                doa_vecs[b, t],       # [S, 3]
                conf_prob[b, t],      # [S]
                cos_thresh=cos_thresh,
                conf_thresh=conf_thresh,
                max_sources=max_sources,
            )
            frame_results.append(keep)
        results.append(frame_results)

    return results
