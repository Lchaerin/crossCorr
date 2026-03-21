#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Dataset Builder
==========================
Builds train / val / test splits of binaural scenes.

Usage
-----
    python build_dataset.py [--output-dir PATH] [--num-train N]
                            [--num-val N] [--num-test N]
                            [--workers N] [--seed N] [--resume]
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import soundfile as sf

# Resolve the package root so we can import sled.dataset even when invoked
# directly as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.dataset.synthesizer import (
    SOFA_PATH, SFX_DIR,
    load_sofa, load_sfx, build_class_map,
    synthesize_scene,
)

# ── Split scene-id offsets ────────────────────────────────────────────────────
SPLIT_OFFSETS = {'train': 0, 'val': 10_000, 'test': 11_000}


# =============================================================================
# Synthetic tone generator (fallback when soud_effects/ is empty)
# =============================================================================

def _generate_synthetic_tones(sfx_dir, fs=48_000, duration=4.0):
    """Generate 10 synthetic sine-wave tones and save as WAV to sfx_dir."""
    freqs = [220, 330, 440, 550, 660, 880, 1100, 1320, 1760, 2200]
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    os.makedirs(sfx_dir, exist_ok=True)
    for freq in freqs:
        fname = f'tone_{freq}hz.wav'
        fpath = os.path.join(sfx_dir, fname)
        if not os.path.exists(fpath):
            wave = 0.5 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
            sf.write(fpath, wave, fs)
    print(f'[SFX] Generated {len(freqs)} synthetic tones in {sfx_dir}')


# =============================================================================
# Worker initialiser / task function (for multiprocessing)
# =============================================================================

_WORKER_STATE: dict = {}


def _worker_init(sofa_path, sfx_dir, class_map):
    """Called once per worker process to load shared resources."""
    hrir_l, hrir_r, azimuths, elevations, fs_hrtf = load_sofa(sofa_path)
    sfx = load_sfx(sfx_dir, int(fs_hrtf))
    _WORKER_STATE['hrir_l']     = hrir_l
    _WORKER_STATE['hrir_r']     = hrir_r
    _WORKER_STATE['azimuths']   = azimuths
    _WORKER_STATE['elevations'] = elevations
    _WORKER_STATE['fs']         = int(fs_hrtf)
    _WORKER_STATE['sfx']        = sfx
    _WORKER_STATE['class_map']  = class_map


def _synthesize_task(args):
    """Top-level function (picklable) executed by each pool worker."""
    name, audio_dir, anno_dir, seed = args
    try:
        synthesize_scene(
            name          = name,
            hrir_l        = _WORKER_STATE['hrir_l'],
            hrir_r        = _WORKER_STATE['hrir_r'],
            azimuths      = _WORKER_STATE['azimuths'],
            elevations    = _WORKER_STATE['elevations'],
            sfx           = _WORKER_STATE['sfx'],
            class_map     = _WORKER_STATE['class_map'],
            output_dir    = audio_dir,   # WAV + JSON go here
            fs            = _WORKER_STATE['fs'],
            seed          = seed,
        )
        # Move annotation .npy files to annotations directory
        for suffix in ('_cls.npy', '_doa.npy', '_loud.npy', '_mask.npy'):
            src = os.path.join(audio_dir, f'{name}{suffix}')
            dst = os.path.join(anno_dir,  f'{name}{suffix}')
            if os.path.exists(src):
                os.replace(src, dst)
        return name, None
    except Exception as exc:
        return name, str(exc)


# =============================================================================
# Build one split
# =============================================================================

def _build_split(split, n_scenes, output_dir, workers, seed, resume):
    """Build all scenes for one split, using multiprocessing."""
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(it, **kw):
            total = kw.get('total', '?')
            print(f'[{split}] generating {total} scenes ...')
            return it

    audio_dir = os.path.join(output_dir, 'audio', split)
    anno_dir  = os.path.join(output_dir, 'annotations', split)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(anno_dir,  exist_ok=True)

    base_id = SPLIT_OFFSETS[split]
    rng     = np.random.RandomState(seed + base_id)

    tasks = []
    for i in range(n_scenes):
        scene_id  = base_id + i
        name      = f'scene_{scene_id:06d}'
        wav_path  = os.path.join(audio_dir, f'{name}.wav')
        if resume and os.path.exists(wav_path):
            continue
        scene_seed = int(rng.randint(0, 2**31 - 1))
        tasks.append((name, audio_dir, anno_dir, scene_seed))

    if not tasks:
        print(f'[{split}] All scenes already exist — skipping.')
        return

    # Prepare shared state for initialiser
    # Load SOFA + sfx in the main process to get the class_map
    hrir_l, hrir_r, azimuths, elevations, fs_hrtf = load_sofa(SOFA_PATH)
    sfx = load_sfx(SFX_DIR, int(fs_hrtf))
    class_map = build_class_map(sfx)

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(
        processes   = max(1, workers),
        initializer = _worker_init,
        initargs    = (SOFA_PATH, SFX_DIR, class_map),
    )

    errors = []
    with _tqdm(total=len(tasks), desc=split) as pbar:
        for name, err in pool.imap_unordered(_synthesize_task, tasks):
            if err:
                errors.append((name, err))
                print(f'\n  [ERROR] {name}: {err}')
            pbar.update(1)

    pool.close()
    pool.join()

    if errors:
        print(f'[{split}] {len(errors)} scene(s) failed.')
    else:
        print(f'[{split}] Done — {len(tasks)} scene(s) synthesised.')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build SLED v3 training dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output-dir', default='./data',
                        help='Root directory for the dataset')
    parser.add_argument('--num-train', type=int, default=100)
    parser.add_argument('--num-val',   type=int, default=20)
    parser.add_argument('--num-test',  type=int, default=10)
    parser.add_argument('--workers',   type=int, default=4,
                        help='Number of parallel worker processes')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--resume',    action='store_true',
                        help='Skip scenes whose WAV already exists')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    meta_dir   = os.path.join(output_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)

    # ── Check / generate SFX ─────────────────────────────────────────────────
    mp3_files = [f for f in os.listdir(SFX_DIR) if f.lower().endswith('.mp3')]
    wav_files = [f for f in os.listdir(SFX_DIR) if f.lower().endswith('.wav')]
    if not mp3_files and not wav_files:
        print('[SFX] soud_effects/ directory is empty — generating synthetic tones.')
        _generate_synthetic_tones(SFX_DIR, fs=48_000, duration=4.0)

    # ── Load resources (main process, for meta) ───────────────────────────────
    print('[SOFA] Loading HRTF ...')
    hrir_l, hrir_r, azimuths, elevations, fs_hrtf = load_sofa(SOFA_PATH)
    fs = int(fs_hrtf)
    print(f'       {len(azimuths)} directions, fs={fs} Hz')

    print('[SFX]  Loading sound effects ...')
    sfx = load_sfx(SFX_DIR, fs)
    print(f'       {len(sfx)} files loaded')

    class_map = build_class_map(sfx)

    # ── Save meta files ───────────────────────────────────────────────────────
    class_map_path = os.path.join(meta_dir, 'class_map.json')
    with open(class_map_path, 'w') as f:
        json.dump(class_map, f, indent=2)
    print(f'[META] class_map → {class_map_path}')

    split_meta = {
        'train': {
            'n_scenes'  : args.num_train,
            'base_id'   : SPLIT_OFFSETS['train'],
            'audio_dir' : os.path.join(output_dir, 'audio', 'train'),
            'anno_dir'  : os.path.join(output_dir, 'annotations', 'train'),
        },
        'val': {
            'n_scenes'  : args.num_val,
            'base_id'   : SPLIT_OFFSETS['val'],
            'audio_dir' : os.path.join(output_dir, 'audio', 'val'),
            'anno_dir'  : os.path.join(output_dir, 'annotations', 'val'),
        },
        'test': {
            'n_scenes'  : args.num_test,
            'base_id'   : SPLIT_OFFSETS['test'],
            'audio_dir' : os.path.join(output_dir, 'audio', 'test'),
            'anno_dir'  : os.path.join(output_dir, 'annotations', 'test'),
        },
    }
    split_meta_path = os.path.join(meta_dir, 'split.json')
    with open(split_meta_path, 'w') as f:
        json.dump(split_meta, f, indent=2)
    print(f'[META] split.json → {split_meta_path}')

    # ── Build splits ──────────────────────────────────────────────────────────
    for split, n in [('train', args.num_train),
                     ('val',   args.num_val),
                     ('test',  args.num_test)]:
        print(f'\n[BUILD] {split} ({n} scenes) ...')
        _build_split(split, n, output_dir, args.workers, args.seed, args.resume)

    print('\nDataset build complete.')


if __name__ == '__main__':
    main()
