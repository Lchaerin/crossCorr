#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — PyTorch Dataset & DataLoader
=======================================
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

# Make the package importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class SLEDDataset(Dataset):
    """Per-scene binaural SLED dataset.

    Parameters
    ----------
    dataset_root  : str   path to the root produced by build_dataset.py
    split         : str   'train', 'val', or 'test'
    window_frames : int   number of annotation frames per sample
                          (256 × 20 ms = 5.12 s)
    augment_scs   : bool  if True and split=='train', apply Stereo Channel Swap

    __getitem__ returns
    -------------------
    {
      'audio'    : [2, window_frames * 960]  float32
      'cls'      : [window_frames, 5]        int64
      'doa'      : [window_frames, 5, 3]     float32
      'loud'     : [window_frames, 5]        float32
      'mask'     : [window_frames, 5]        bool
      'scene_id' : str
    }
    """

    HOP_SAMPLES = 960   # samples per annotation frame (20 ms @ 48 kHz)
    MAX_SLOTS   = 3

    def __init__(self, dataset_root: str, split: str,
                 window_frames: int = 256, augment_scs: bool = True,
                 min_loudness_db: float = -60.0):
        super().__init__()
        self.dataset_root    = dataset_root
        self.split           = split
        self.window_frames   = window_frames
        self.augment_scs     = augment_scs and (split == 'train')
        self.min_loudness_db = min_loudness_db
        self._max_sources    = self.MAX_SLOTS   # default: no restriction

        # Load split metadata
        split_meta_path = os.path.join(dataset_root, 'meta', 'split.json')
        with open(split_meta_path, 'r') as f:
            split_meta = json.load(f)

        if split not in split_meta:
            raise ValueError(f"Split '{split}' not found in {split_meta_path}")

        info = split_meta[split]
        self.audio_dir = info['audio_dir']
        self.anno_dir  = info['anno_dir']
        n_scenes       = info['n_scenes']
        base_id        = info['base_id']

        # Build file list (only include scenes that are fully synthesised)
        self.scene_ids = []
        for i in range(n_scenes):
            scene_id = base_id + i
            name     = f'scene_{scene_id:06d}'
            wav_path = os.path.join(self.audio_dir, f'{name}.wav')
            cls_path = os.path.join(self.anno_dir,  f'{name}_cls.npy')
            if os.path.exists(wav_path) and os.path.exists(cls_path):
                self.scene_ids.append(name)

        if not self.scene_ids:
            raise RuntimeError(
                f"No complete scenes found for split '{split}' in {dataset_root}. "
                "Run build_dataset.py first."
            )

    # ─────────────────────────────────────────────────────────────────────────

    def set_max_sources(self, max_sources: int) -> None:
        """커리큘럼 학습용: 이 epoch에서 사용할 최대 활성 음원 수 제한.

        max_sources=1이면 프레임당 1개만 남기고 나머지 마스크 제거.
        max_sources=2이면 최대 2개 유지.
        max_sources=3(기본)이면 제한 없음.
        """
        self._max_sources = max(1, int(max_sources))

    def __len__(self) -> int:
        return len(self.scene_ids)

    # ─────────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        import soundfile as sf

        name = self.scene_ids[idx]

        # Load stereo waveform
        wav_path = os.path.join(self.audio_dir, f'{name}.wav')
        audio_np, _ = sf.read(wav_path, dtype='float32', always_2d=True)
        # audio_np: [N_samples, 2] → [2, N_samples]
        audio_np = audio_np.T

        # Load annotation arrays
        cls_arr  = np.load(os.path.join(self.anno_dir, f'{name}_cls.npy'))   # [T, 5] int16
        doa_arr  = np.load(os.path.join(self.anno_dir, f'{name}_doa.npy'))   # [T, 5, 3] float16
        loud_arr = np.load(os.path.join(self.anno_dir, f'{name}_loud.npy'))  # [T, 5] float16
        mask_arr = np.load(os.path.join(self.anno_dir, f'{name}_mask.npy'))  # [T, 5] bool

        T_total = cls_arr.shape[0]

        # Random window selection
        max_start = max(0, T_total - self.window_frames)
        if max_start > 0:
            t_start = np.random.randint(0, max_start)
        else:
            t_start = 0
        t_end = t_start + self.window_frames

        # Slice annotations (pad if scene is shorter than window)
        def _slice_pad_2d(arr, t_start, t_end, pad_val):
            sliced = arr[t_start:t_end]
            pad_len = self.window_frames - sliced.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len,) + sliced.shape[1:]
                padding   = np.full(pad_shape, pad_val, dtype=sliced.dtype)
                sliced    = np.concatenate([sliced, padding], axis=0)
            return sliced

        cls_w  = _slice_pad_2d(cls_arr,  t_start, t_end, -1)
        doa_w  = _slice_pad_2d(doa_arr,  t_start, t_end,  0)
        loud_w = _slice_pad_2d(loud_arr, t_start, t_end, -80)
        mask_w = _slice_pad_2d(mask_arr, t_start, t_end,  False)

        # Suppress slots that are below the loudness threshold.
        # Even if the JSON says a source is present, if it's inaudible
        # the model should treat it as inactive.
        too_quiet = loud_w < self.min_loudness_db   # [T, 5] bool
        mask_w    = mask_w & ~too_quiet
        cls_w     = cls_w.copy()
        cls_w[too_quiet] = -1   # sentinel: inactive

        # Curriculum: if _max_sources < MAX_SLOTS, randomly suppress extras.
        if self._max_sources < self.MAX_SLOTS:
            for t in range(mask_w.shape[0]):
                active_idx = np.where(mask_w[t])[0]
                if len(active_idx) > self._max_sources:
                    n_suppress = len(active_idx) - self._max_sources
                    suppress = np.random.choice(active_idx, n_suppress,
                                                replace=False)
                    mask_w[t, suppress] = False
                    cls_w[t, suppress]  = -1

        # Slice audio
        s_start = t_start * self.HOP_SAMPLES
        s_end   = t_end   * self.HOP_SAMPLES
        n_audio = self.window_frames * self.HOP_SAMPLES
        total_audio = audio_np.shape[1]

        if s_end <= total_audio:
            audio_w = audio_np[:, s_start:s_end]
        else:
            audio_w = audio_np[:, s_start:total_audio]
            pad_len = n_audio - audio_w.shape[1]
            if pad_len > 0:
                audio_w = np.concatenate(
                    [audio_w, np.zeros((2, pad_len), dtype=np.float32)],
                    axis=1
                )

        # Convert to torch tensors
        audio_t = torch.from_numpy(audio_w.astype(np.float32))   # [2, N]
        cls_t   = torch.from_numpy(cls_w.astype(np.int64))        # [T, 5]
        doa_t   = torch.from_numpy(doa_w.astype(np.float32))      # [T, 5, 3]
        loud_t  = torch.from_numpy(loud_w.astype(np.float32))     # [T, 5]
        mask_t  = torch.from_numpy(mask_w.astype(bool))           # [T, 5]

        # ── Stereo Channel Swap augmentation ─────────────────────────────────
        if self.augment_scs and torch.rand(1).item() < 0.5:
            # Swap L and R
            audio_t = audio_t.flip(0)
            # Negate azimuth: doa = (x, y, z), y encodes azimuth in SLED
            # CW convention: negating y negates azimuth
            doa_t = doa_t.clone()
            doa_t[..., 1] = -doa_t[..., 1]

        return {
            'audio'   : audio_t,
            'cls'     : cls_t,
            'doa'     : doa_t,
            'loud'    : loud_t,
            'mask'    : mask_t,
            'scene_id': name,
        }


# =============================================================================
# DataLoader factory
# =============================================================================

def build_mixed_dataloader(
    dataset_configs: list,
    split: str,
    batch_size: int = 8,
    window_frames: int = 256,
    augment_scs: bool = True,
    num_workers: int = 4,
    min_loudness_db: float = -60.0,
    epoch_size: int | None = None,
    val_size: int | None = None,
    **kwargs,
) -> DataLoader:
    """여러 데이터셋을 비율에 따라 섞는 DataLoader.

    Parameters
    ----------
    dataset_configs : list of dict
        [{'root': './data_custom_hrtf', 'weight': 0.7},
         {'root': './data_mrs_mix',     'weight': 0.3}]
        weight는 정규화 불필요 (자동 처리).
    epoch_size : int or None
        train 에폭당 총 샘플 수. None이면 두 데이터셋 합계(N1+N2).
    val_size : int or None
        val/test 에폭당 총 샘플 수. None이면 두 데이터셋 합계(N1+N2).
        가중치 비율에 따라 각 데이터셋에서 proportional하게 샘플링.
    """
    datasets = []
    for cfg in dataset_configs:
        ds = SLEDDataset(
            dataset_root    = cfg['root'],
            split           = split,
            window_frames   = window_frames,
            augment_scs     = augment_scs,
            min_loudness_db = min_loudness_db,
        )
        datasets.append((ds, float(cfg.get('weight', 1.0))))

    pin_memory = kwargs.pop('pin_memory', True)

    weights_raw = [w for _, w in datasets]
    total_w     = sum(weights_raw)
    norm_w      = [w / total_w for w in weights_raw]

    combined = ConcatDataset([ds for ds, _ in datasets])

    sample_weights = []
    for (ds, _), nw in zip(datasets, norm_w):
        per_sample = nw / len(ds)
        sample_weights.extend([per_sample] * len(ds))

    if split == 'train':
        n_samples = epoch_size if epoch_size is not None else len(combined)
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = n_samples,
            replacement = True,
        )
        return DataLoader(
            combined,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,
            **kwargs,
        )
    else:
        # val/test: 동일한 가중치 비율로 고정 seed 샘플링 (결정론적)
        n_val = val_size if val_size is not None else len(combined)
        n_val = min(n_val, len(combined))
        g = torch.Generator()
        g.manual_seed(0)
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = n_val,
            replacement = False,
            generator   = g,
        )
        return DataLoader(
            combined,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = False,
            **kwargs,
        )


def build_dataloader(
    dataset_root: str,
    split: str,
    batch_size: int = 8,
    window_frames: int = 256,
    augment_scs: bool = True,
    num_workers: int = 4,
    min_loudness_db: float = -60.0,
    epoch_size: int | None = None,
    **kwargs,
) -> DataLoader:
    """Build a DataLoader for the given split.

    Parameters
    ----------
    dataset_root  : root produced by build_dataset.py
    split         : 'train', 'val', or 'test'
    batch_size    : mini-batch size
    window_frames : frames per sample (256 × 20 ms = 5.12 s)
    augment_scs   : Stereo Channel Swap augmentation (train only)
    num_workers   : DataLoader worker count
    epoch_size    : train 에폭당 샘플 수. None이면 데이터셋 전체.
    **kwargs      : forwarded to DataLoader

    Returns
    -------
    torch.utils.data.DataLoader
    """
    from torch.utils.data import RandomSampler
    dataset = SLEDDataset(
        dataset_root    = dataset_root,
        split           = split,
        window_frames   = window_frames,
        augment_scs     = augment_scs,
        min_loudness_db = min_loudness_db,
    )
    pin_memory = kwargs.pop('pin_memory', True)
    kwargs.pop('shuffle', None)  # sampler와 shuffle 동시 사용 방지

    if split == 'train' and epoch_size is not None:
        sampler = RandomSampler(dataset, replacement=True, num_samples=epoch_size)
        return DataLoader(
            dataset,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == 'train'),
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = (split == 'train'),
        **kwargs,
    )
