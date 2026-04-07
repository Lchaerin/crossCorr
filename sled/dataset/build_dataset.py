#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Dataset Builder
==========================
Builds train / val / test splits of binaural scenes with strict data separation:

  FSD50K clips
    train  ← dev.csv  split='train'   (~36,796 clips)
    val    ← dev.csv  split='val'     (~4,170  clips)
    test   ← eval.csv (all)           (~10,231 clips)

  SRIR rooms
    train  ← 7 rooms  (bomb_shelter, gym, pb132, pc226, sa203, sc203, se203)
    val    ← 2 rooms  (tb103, tc352)   ← unseen during training
    test   ← 2 rooms  (tb103, tc352)   ← same held-out set

  HRTF subjects
    all splits ← random from 140 p*.sofa  (no split; ears are listener-generic)

Usage
-----
    # Full dataset (16 workers)
    python -m sled.dataset.build_dataset --output-dir ./data --workers 16

    # Quick smoke test
    python -m sled.dataset.build_dataset --output-dir ./data_test \\
        --num-train 10 --num-val 2 --num-test 2 --workers 1

    # Resume interrupted build
    python -m sled.dataset.build_dataset --output-dir ./data --workers 16 --resume

    # Custom paths
    python -m sled.dataset.build_dataset --output-dir ./data \\
        --hrtf-dir ./hrtf --sfx-dir ./soud_effects \\
        --srir-dir ./sources/TAU_SRIR/TAU-SRIR_DB \\
        --fsd50k-gt-dir ./sources/FSD50K/FSD50K.ground_truth \\
        --workers 16
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.dataset.synthesizer import (
    HRTF_DIR, SFX_DIR, SRIR_DIR, FSD50K_GT_DIR,
    SRIR_TRAIN_ROOMS, SRIR_EVAL_ROOMS,
    scan_sofa_paths, scan_sfx_split_paths, scan_sfx_paths,
    load_sfx_from_paths, build_class_map,
    synthesize_scene,
)

# ── Split scene-id offsets ────────────────────────────────────────────────────
SPLIT_OFFSETS = {'train': 0, 'val': 20_000, 'test': 22_500}

# ── SRIR rooms per split ──────────────────────────────────────────────────────
_SRIR_ROOMS_PER_SPLIT = {
    'train': SRIR_TRAIN_ROOMS,
    'val':   SRIR_EVAL_ROOMS,
    'test':  SRIR_EVAL_ROOMS,
}


# =============================================================================
# Helpers
# =============================================================================

def _generate_synthetic_tones(sfx_dir, fs=48_000, duration=4.0):
    """Generate 10 synthetic sine-wave tones (fallback when sfx_dir is empty)."""
    freqs = [220, 330, 440, 550, 660, 880, 1100, 1320, 1760, 2200]
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    os.makedirs(sfx_dir, exist_ok=True)
    for freq in freqs:
        fpath = os.path.join(sfx_dir, f'tone_{freq}hz.wav')
        if not os.path.exists(fpath):
            sf.write(fpath, 0.5 * np.sin(2.0 * np.pi * freq * t).astype(np.float32), fs)
    print(f'[SFX] Generated {len(freqs)} synthetic tones in {sfx_dir}')


def _read_sofa_fs(sofa_path):
    """Read sample rate from a SOFA file without loading all data."""
    try:
        import netCDF4 as _nc
        ds = _nc.Dataset(sofa_path, 'r')
        fs = int(np.asarray(ds.variables['Data.SamplingRate'][:])[0])
        ds.close()
        return fs
    except Exception:
        import h5py as _h5
        with _h5.File(sofa_path, 'r') as f:
            return int(np.array(f['Data.SamplingRate']).flat[0])


# =============================================================================
# Worker initialiser / task function
# =============================================================================

_WORKER_STATE: dict = {}


def _worker_init(sofa_paths, sfx_paths, class_map, max_sfx_files,
                 srir_dir, srir_rooms, fs):
    """Called once per worker process.  HRTF subject chosen per scene."""
    sfx = load_sfx_from_paths(sfx_paths, fs,
                               max_files=max_sfx_files, seed=os.getpid())
    _WORKER_STATE['sofa_paths'] = sofa_paths
    _WORKER_STATE['fs']         = fs
    _WORKER_STATE['sfx']        = sfx
    _WORKER_STATE['class_map']  = class_map
    _WORKER_STATE['srir_dir']   = srir_dir
    _WORKER_STATE['srir_rooms'] = srir_rooms


def _synthesize_task(args):
    name, audio_dir, anno_dir, seed = args
    try:
        synthesize_scene(
            name       = name,
            sofa_paths = _WORKER_STATE['sofa_paths'],
            sfx        = _WORKER_STATE['sfx'],
            class_map  = _WORKER_STATE['class_map'],
            output_dir = audio_dir,
            fs         = _WORKER_STATE['fs'],
            srir_dir   = _WORKER_STATE['srir_dir'],
            srir_rooms = _WORKER_STATE['srir_rooms'],
            seed       = seed,
        )
        for suffix in ('_cls.npy', '_doa.npy', '_loud.npy', '_mask.npy'):
            src = os.path.join(audio_dir, f'{name}{suffix}')
            dst = os.path.join(anno_dir,  f'{name}{suffix}')
            if os.path.exists(src):
                os.replace(src, dst)
        return name, None
    except Exception:
        import traceback
        return name, traceback.format_exc()


# =============================================================================
# Build one split
# =============================================================================

def _build_split(split, n_scenes, output_dir, workers, seed, resume,
                 sofa_paths, sfx_paths, class_map, max_sfx_files,
                 srir_dir, srir_rooms, fs):
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        class _tqdm:
            def __init__(self, it=None, **kw):
                print(f'[{split}] generating {kw.get("total","?")} scenes ...')
                self._it = it
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __iter__(self): return iter(self._it)
            def update(self, n=1): pass

    audio_dir = os.path.join(output_dir, 'audio', split)
    anno_dir  = os.path.join(output_dir, 'annotations', split)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(anno_dir,  exist_ok=True)

    base_id = SPLIT_OFFSETS[split]
    rng     = np.random.RandomState(seed + base_id)

    tasks = []
    for i in range(n_scenes):
        scene_id   = base_id + i
        name       = f'scene_{scene_id:06d}'
        wav_path   = os.path.join(audio_dir, f'{name}.wav')
        if resume and os.path.exists(wav_path):
            continue
        tasks.append((name, audio_dir, anno_dir, int(rng.randint(0, 2**31 - 1))))

    if not tasks:
        print(f'[{split}] All scenes already exist — skipping.')
        return

    ctx  = mp.get_context('spawn')
    pool = ctx.Pool(
        processes   = max(1, workers),
        initializer = _worker_init,
        initargs    = (sofa_paths, sfx_paths, class_map,
                       max_sfx_files, srir_dir, srir_rooms, fs),
    )

    errors = []
    with _tqdm(total=len(tasks), desc=split) as pbar:
        for name, err in pool.imap_unordered(_synthesize_task, tasks):
            if err:
                errors.append((name, err))
                print(f'\n  [ERROR] {name}:\n{err}')
            pbar.update(1)

    pool.close()
    pool.join()

    n_err = len(errors)
    if n_err:
        print(f'[{split}] {n_err} scene(s) failed.')
    else:
        print(f'[{split}] Done — {len(tasks)} scene(s) synthesised.')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build SLED v3 binaural training dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output-dir',      default='./data')
    parser.add_argument('--hrtf-dir',        default=HRTF_DIR,
                        help='Directory containing p*.sofa HRTF files')
    parser.add_argument('--sfx-dir',         default=SFX_DIR,
                        help='Directory containing sound effect clips (class/ layout)')
    parser.add_argument('--srir-dir',        default=SRIR_DIR,
                        help='Path to TAU-SRIR_DB directory')
    parser.add_argument('--fsd50k-gt-dir',   default=FSD50K_GT_DIR,
                        help='Path to FSD50K.ground_truth/ (for split-aware clip selection)')
    parser.add_argument('--num-train',       type=int, default=20_000)
    parser.add_argument('--num-val',         type=int, default=1_500)
    parser.add_argument('--num-test',        type=int, default=500)
    parser.add_argument('--workers',         type=int, default=4)
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--resume',          action='store_true',
                        help='Skip scenes whose WAV already exists')
    parser.add_argument('--max-sfx-files',   type=int, default=500,
                        help='Max SFX clips loaded per worker (0 = all)')
    parser.add_argument('--no-split-sfx',    action='store_true',
                        help='Disable FSD50K split filtering (use all clips for every split)')
    args = parser.parse_args()

    output_dir    = os.path.abspath(args.output_dir)
    max_sfx_files = args.max_sfx_files if args.max_sfx_files > 0 else None
    os.makedirs(os.path.join(output_dir, 'meta'), exist_ok=True)

    # ── HRTF ─────────────────────────────────────────────────────────────────
    print('[HRTF] Scanning SOFA files ...')
    sofa_paths = scan_sofa_paths(args.hrtf_dir)
    if not sofa_paths:
        raise FileNotFoundError(f'No p*.sofa files in {args.hrtf_dir}')
    fs = _read_sofa_fs(sofa_paths[0])
    print(f'       {len(sofa_paths)} subjects, fs={fs} Hz')

    # ── SFX — per-split clip sets ─────────────────────────────────────────────
    print('[SFX]  Scanning sound effects ...')
    gt_dir = args.fsd50k_gt_dir

    use_split = (not args.no_split_sfx
                 and os.path.isdir(gt_dir)
                 and os.path.exists(os.path.join(gt_dir, 'dev.csv')))

    if use_split:
        sfx_paths_per_split = {
            'train': scan_sfx_split_paths(args.sfx_dir, gt_dir, 'train'),
            'val':   scan_sfx_split_paths(args.sfx_dir, gt_dir, 'val'),
            'test':  scan_sfx_split_paths(args.sfx_dir, gt_dir, 'test'),
        }
        for sp, paths in sfx_paths_per_split.items():
            print(f'       {sp:5s}: {len(paths):6,} clips')
    else:
        print('       FSD50K ground-truth CSV not found — using all clips for every split')
        all_paths = scan_sfx_paths(args.sfx_dir)
        sfx_paths_per_split = {'train': all_paths, 'val': all_paths, 'test': all_paths}
        print(f'       total: {len(all_paths):,} clips')

    # Build unified class_map from all clips so IDs are consistent across splits
    all_sfx_paths = scan_sfx_paths(args.sfx_dir)
    if not all_sfx_paths:
        print('[SFX]  soud_effects/ is empty — generating synthetic tones.')
        _generate_synthetic_tones(args.sfx_dir, fs=fs, duration=4.0)
        all_sfx_paths = scan_sfx_paths(args.sfx_dir)
        sfx_paths_per_split = {'train': all_sfx_paths,
                                'val':   all_sfx_paths,
                                'test':  all_sfx_paths}
    class_map = build_class_map(all_sfx_paths)

    # ── SRIR ─────────────────────────────────────────────────────────────────
    print(f'[SRIR] train rooms ({len(SRIR_TRAIN_ROOMS)}): {SRIR_TRAIN_ROOMS}')
    print(f'       eval  rooms ({len(SRIR_EVAL_ROOMS)}): {SRIR_EVAL_ROOMS}')

    # ── Meta files ────────────────────────────────────────────────────────────
    meta_dir = os.path.join(output_dir, 'meta')
    with open(os.path.join(meta_dir, 'class_map.json'), 'w') as f:
        json.dump(class_map, f, indent=2)

    split_cfg = {
        sp: {
            'n_scenes' : getattr(args, f'num_{sp}'),
            'base_id'  : SPLIT_OFFSETS[sp],
            'srir_rooms': _SRIR_ROOMS_PER_SPLIT[sp],
            'audio_dir': os.path.join(output_dir, 'audio', sp),
            'anno_dir' : os.path.join(output_dir, 'annotations', sp),
        }
        for sp in ('train', 'val', 'test')
    }
    with open(os.path.join(meta_dir, 'split.json'), 'w') as f:
        json.dump(split_cfg, f, indent=2)
    print(f'[META] Written to {meta_dir}/')

    # ── Build splits ──────────────────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        n = getattr(args, f'num_{split}')
        print(f'\n[BUILD] {split} ({n} scenes) ...')
        _build_split(
            split        = split,
            n_scenes     = n,
            output_dir   = output_dir,
            workers      = args.workers,
            seed         = args.seed,
            resume       = args.resume,
            sofa_paths   = sofa_paths,
            sfx_paths    = sfx_paths_per_split[split],
            class_map    = class_map,
            max_sfx_files= max_sfx_files,
            srir_dir     = args.srir_dir,
            srir_rooms   = _SRIR_ROOMS_PER_SPLIT[split],
            fs           = fs,
        )

    print('\nDataset build complete.')


if __name__ == '__main__':
    main()
