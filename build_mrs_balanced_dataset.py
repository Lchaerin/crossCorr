#!/usr/bin/env python3
"""
build_mrs_balanced_dataset.py
==============================
MRS binaural 녹음들을 azimuth-balanced 방식으로 혼합하여 multi-source 합성 데이터셋을 생성합니다.
출력 포맷은 data_mrs_mix와 완전히 동일합니다.

균형화 방법
-----------
- 전체 세그먼트를 azimuth 구간(기본 12 bins × 30°)별로 분류
- 세그먼트 가중치 = 1 / 해당 bin 내 세그먼트 수
  → 모든 bin이 동일한 확률(1/N_BINS)로 선택됨
- 씬 생성 시 가중치에 따라 세그먼트를 샘플링 (서로 다른 sound 디렉터리)

Usage
-----
    python build_mrs_balanced_dataset.py \\
        --mrs-root ./MRSAudio/MRSLife/MRSSound \\
        --out-dir  ./data_mrs_balanced \\
        --n-train  6000 \\
        --n-val    750  \\
        --n-test   250  \\
        --n-bins   12   \\
        --seed     42
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

MRS_TO_FSD50K_MAP: dict[str, int] = {
    'bell'                    :  11,
    'handbell'                :  11,
    'stickbell'               :  34,
    'gong'                    :  87,
    'ClashCymbals'            :  48,
    'chinesecymbals'          :  57,
    'triangle'                :  34,
    'clap'                    :  39,
    'slitdrum'                : 111,
    'woodenfish'              : 111,
    'MultiPitchPercussionTube': 111,
    'WoodenClapper'           :  39,
    'woodenclapper'           :  39,
    'rattle'                  : 137,
    'maracas'                 : 137,
    'woodenshaker'            : 137,
    'tambourine'              : 165,
    'rotatingclapperboard'    : 135,
    'rainstick'               : 133,
    'thundertube'             : 170,
    'birdwhistle'             :  36,
    'whistle'                 : 194,
    'fanwithpaper'            : 190,
    'hairdryer'               : 113,
    'squeakingrubberchicken'  : 160,
    'Tear off tape'           : 123,
    'toy car'                 :  26,
    'toy train'               : 177,
    'toy train 8'             : 177,
}

SR          = 48_000
HOP         = 960
T_SCENE     = 1500
N_SAMPLES   = T_SCENE * HOP
MAX_SLOTS   = 3
MIN_LOUD_DB = -60.0
TARGET_RMS  = 0.05
SRC_START_MIN = 0.0
SRC_START_MAX = 20.0


# =============================================================================
# 세그먼트 풀 로딩 + azimuth 가중치 계산
# =============================================================================

def compute_mean_az_deg(npy_path: str) -> float:
    """npy 파일에서 세그먼트의 circular mean azimuth(도) 계산.

    SLED CW 규약: az = atan2(right, fwd), 0°=정면, 90°=오른쪽
    """
    npy   = np.load(npy_path)
    right = npy[:, 0]
    fwd   = npy[:, 1]
    az_rad = np.arctan2(right, fwd)
    mean_az = np.arctan2(np.mean(np.sin(az_rad)), np.mean(np.cos(az_rad)))
    return float(np.degrees(mean_az))


def load_balanced_pool(mrs_root: str, sound_dirs: list[str],
                       n_bins: int = 12) -> list[dict]:
    """세그먼트 목록 로딩 + azimuth bin 가중치 계산.

    Returns
    -------
    flat list of seg_dict:
        wav_path, npy_path, class_id, event, start_ms, stop_ms,
        snd_dir, mean_az_deg, az_bin, weight
    """
    mrs_base = os.path.dirname(os.path.dirname(mrs_root))
    bin_edges = np.linspace(-180, 180, n_bins + 1)

    flat: list[dict] = []

    print(f'  세그먼트 로딩 및 azimuth 계산 중 ({len(sound_dirs)} sounds)...')
    for snd in sound_dirs:
        snd_dir   = os.path.join(mrs_root, snd)
        meta_path = os.path.join(snd_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta_list = json.load(f)

        for seg in meta_list:
            event = seg.get('event', '')
            if event not in MRS_TO_FSD50K_MAP:
                continue

            npy_path = os.path.join(mrs_base, seg['pos_fn'])
            wav_path = os.path.join(mrs_base, seg['wav_fn'])
            if not os.path.exists(npy_path) or not os.path.exists(wav_path):
                continue

            start_ms = seg['start'] * 1000.0
            stop_ms  = seg['stop']  * 1000.0
            if stop_ms - start_ms < 1000.0:
                continue

            mean_az = compute_mean_az_deg(npy_path)
            az_bin  = int(np.clip(np.digitize(mean_az, bin_edges) - 1, 0, n_bins - 1))

            flat.append({
                'wav_path' : wav_path,
                'npy_path' : npy_path,
                'class_id' : MRS_TO_FSD50K_MAP[event],
                'event'    : event,
                'start_ms' : start_ms,
                'stop_ms'  : stop_ms,
                'snd_dir'  : snd,
                'mean_az'  : mean_az,
                'az_bin'   : az_bin,
                'weight'   : 0.0,   # 아래에서 채움
            })

    # bin별 카운트 → weight = 1 / bin_count
    bin_counts = np.zeros(n_bins, dtype=int)
    for s in flat:
        bin_counts[s['az_bin']] += 1

    print(f'  Azimuth bin 분포 ({n_bins} bins, 30°씩):')
    for i in range(n_bins):
        lo = bin_edges[i]; hi = bin_edges[i + 1]
        print(f'    {lo:6.0f}~{hi:6.0f}°: {bin_counts[i]:5d} segs')

    for s in flat:
        cnt = bin_counts[s['az_bin']]
        s['weight'] = 1.0 / cnt if cnt > 0 else 0.0

    return flat


# =============================================================================
# 소스 로딩 (원본과 동일)
# =============================================================================

def load_source(seg: dict):
    start_ms = seg['start_ms']
    stop_ms  = seg['stop_ms']
    dur_ms   = stop_ms - start_ms

    s_start = int(start_ms * SR / 1000)
    s_stop  = int(stop_ms  * SR / 1000)
    try:
        audio_np, _ = sf.read(
            seg['wav_path'], start=s_start, stop=s_stop,
            dtype='float32', always_2d=True,
        )
    except Exception:
        n_exp = int(dur_ms * SR / 1000)
        audio_np = np.zeros((n_exp, 2), dtype=np.float32)

    audio_np = audio_np.T  # [2, N_seg]

    npy = np.load(seg['npy_path'])
    npy_times   = npy[:, 3]
    npy_right   = npy[:, 0]
    npy_forward = npy[:, 1]
    npy_up      = npy[:, 2]

    T_seg = max(1, int(dur_ms / 20.0))
    frame_times_ms = start_ms + (np.arange(T_seg) + 0.5) * 20.0
    frame_times_cl = np.clip(frame_times_ms, npy_times[0], npy_times[-1])

    right_f   = np.interp(frame_times_cl, npy_times, npy_right)
    forward_f = np.interp(frame_times_cl, npy_times, npy_forward)
    up_f      = np.interp(frame_times_cl, npy_times, npy_up)

    doa_raw = np.stack([forward_f, right_f, up_f], axis=1)
    norm    = np.linalg.norm(doa_raw, axis=1, keepdims=True)
    norm    = np.where(norm < 1e-8, 1.0, norm)
    doa_seq = (doa_raw / norm).astype(np.float32)

    return audio_np, doa_seq, frame_times_ms


# =============================================================================
# 균형 소스 샘플링
# =============================================================================

def sample_sources_balanced(flat_segs: list[dict], n_src: int,
                             rng: np.random.Generator) -> list[dict] | None:
    """가중치 기반으로 n_src개 세그먼트를 서로 다른 sound 디렉터리에서 선택.

    각 세그먼트의 weight = 1 / (해당 azimuth bin 내 세그먼트 수)
    → 모든 방위각 bin이 동일한 확률로 선택됨.
    """
    weights = np.array([s['weight'] for s in flat_segs], dtype=np.float64)
    mask    = np.ones(len(flat_segs), dtype=bool)
    chosen  = []
    used_dirs: set[str] = set()

    for _ in range(n_src):
        w = weights * mask
        total_w = w.sum()
        if total_w < 1e-12:
            return None
        p = w / total_w
        idx = int(rng.choice(len(flat_segs), p=p))

        seg     = flat_segs[idx]
        snd_dir = seg['snd_dir']
        chosen.append(seg)
        used_dirs.add(snd_dir)

        # 선택된 sound 디렉터리의 모든 세그먼트를 마스킹
        for i, s in enumerate(flat_segs):
            if s['snd_dir'] == snd_dir:
                mask[i] = False

    return chosen if len(chosen) == n_src else None


# =============================================================================
# 씬 생성 (원본과 동일)
# =============================================================================

def build_scene(sources: list[dict], scene_id: int, rng: np.random.Generator) -> dict:
    n_src     = len(sources)
    audio_mix = np.zeros((2, N_SAMPLES), dtype=np.float32)
    cls_arr   = np.full((T_SCENE, MAX_SLOTS), -1,    dtype=np.int16)
    doa_arr   = np.zeros((T_SCENE, MAX_SLOTS, 3),    dtype=np.float32)
    loud_arr  = np.full((T_SCENE, MAX_SLOTS), -80.0, dtype=np.float32)
    mask_arr  = np.zeros((T_SCENE, MAX_SLOTS),       dtype=bool)
    events_json = []

    for slot, seg in enumerate(sources):
        audio_src, doa_seq, _ = load_source(seg)
        T_seg = doa_seq.shape[0]
        N_seg = audio_src.shape[1]

        rms = float(np.sqrt(np.mean(audio_src ** 2)))
        audio_src = audio_src * (TARGET_RMS / rms) if rms > 1e-8 else audio_src * TARGET_RMS

        gain_db  = float(rng.uniform(-6.0, 6.0))
        gain     = 10.0 ** (gain_db / 20.0)
        audio_src = audio_src * gain

        max_start_s   = max(0.0, SRC_START_MAX - (T_seg * 0.02))
        scene_start_s = float(rng.uniform(SRC_START_MIN, max(SRC_START_MIN, max_start_s)))
        scene_start_smp = int(scene_start_s * SR)
        scene_end_smp   = min(scene_start_smp + N_seg, N_SAMPLES)
        actual_len_smp  = scene_end_smp - scene_start_smp

        audio_mix[:, scene_start_smp:scene_end_smp] += audio_src[:, :actual_len_smp]

        scene_start_frame = scene_start_smp // HOP
        actual_len_frames = min(T_seg, (actual_len_smp + HOP - 1) // HOP)
        scene_end_frame   = min(scene_start_frame + actual_len_frames, T_SCENE)

        for t_scene in range(scene_start_frame, scene_end_frame):
            t_local = t_scene - scene_start_frame
            if t_local >= T_seg:
                break
            cls_arr[t_scene, slot] = seg['class_id']
            doa_arr[t_scene, slot] = doa_seq[t_local]
            s0 = t_local * HOP
            s1 = min(s0 + HOP, audio_src.shape[1])
            frame_rms = float(np.sqrt(np.mean(audio_src[:, s0:s1] ** 2))) if s1 > s0 else 0.0
            loud_db = 20.0 * np.log10(frame_rms + 1e-9)
            loud_arr[t_scene, slot] = float(np.clip(loud_db, -80.0, 0.0))
            mask_arr[t_scene, slot] = loud_db > MIN_LOUD_DB

        events_json.append({
            'file'        : f"MRS/{seg['event']}",
            'start_sample': int(scene_start_smp),
            'end_sample'  : int(scene_end_smp),
            'start_time'  : round(scene_start_smp / SR, 4),
            'end_time'    : round(scene_end_smp   / SR, 4),
            'azimuth'     : None,
            'elevation'   : None,
            'gain'        : round(float(gain), 4),
            'source_event': seg['event'],
            'class_id'    : int(seg['class_id']),
        })

    peak = float(np.abs(audio_mix).max())
    if peak > 0.95:
        audio_mix = audio_mix * (0.95 / peak)
        scale_db  = 20.0 * np.log10(0.95 / peak)
        loud_arr  = np.clip(loud_arr + scale_db, -80.0, 0.0)
        mask_arr  = loud_arr > MIN_LOUD_DB
        for sl in range(MAX_SLOTS):
            inactive = cls_arr[:, sl] == -1
            loud_arr[inactive, sl] = -80.0
            mask_arr[inactive, sl] = False

    scene_name = f'scene_{scene_id:06d}'
    meta_json  = {
        'scene_name'  : scene_name,
        'duration_sec': round(T_SCENE * HOP / SR, 4),
        'sample_rate' : SR,
        'hop_samples' : HOP,
        'n_frames'    : T_SCENE,
        'max_slots'   : MAX_SLOTS,
        'audio_file'  : f'{scene_name}.wav',
        'num_events'  : n_src,
        'events'      : events_json,
    }
    return {
        'audio': audio_mix,
        'cls'  : cls_arr,
        'doa'  : doa_arr,
        'loud' : loud_arr,
        'mask' : mask_arr,
        'meta' : meta_json,
    }


def save_scene(scene_data: dict, audio_dir: str, anno_dir: str):
    name = scene_data['meta']['scene_name']
    sf.write(os.path.join(audio_dir, f'{name}.wav'),
             scene_data['audio'].T.astype(np.float32), SR, subtype='FLOAT')
    with open(os.path.join(audio_dir, f'{name}.json'), 'w') as f:
        json.dump(scene_data['meta'], f, indent=2)
    np.save(os.path.join(anno_dir, f'{name}_cls.npy'),  scene_data['cls'].astype(np.int16))
    np.save(os.path.join(anno_dir, f'{name}_doa.npy'),  scene_data['doa'].astype(np.float16))
    np.save(os.path.join(anno_dir, f'{name}_loud.npy'), scene_data['loud'].astype(np.float16))
    np.save(os.path.join(anno_dir, f'{name}_mask.npy'), scene_data['mask'])


# =============================================================================
# 스플릿 생성
# =============================================================================

def build_split(flat_segs: list[dict],
                n_scenes: int, base_id: int,
                audio_dir: str, anno_dir: str,
                rng: np.random.Generator,
                src_dist: tuple[float, float, float] = (0.2, 0.4, 0.4)):
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(anno_dir,  exist_ok=True)

    n_per_src = [
        int(n_scenes * src_dist[0]),
        int(n_scenes * src_dist[1]),
        n_scenes - int(n_scenes * src_dist[0]) - int(n_scenes * src_dist[1]),
    ]
    n_src_list = [1] * n_per_src[0] + [2] * n_per_src[1] + [3] * n_per_src[2]
    rng.shuffle(n_src_list)

    # sound 디렉터리 수 확인 (n_src=3이면 최소 3개 필요)
    n_dirs = len(set(s['snd_dir'] for s in flat_segs))
    if n_dirs < 3:
        print(f'  WARNING: sound 디렉터리 {n_dirs}개뿐, n_src 최대 {n_dirs}로 제한')
        n_src_list = [min(n, n_dirs) for n in n_src_list]

    skipped = 0
    scene_idx = 0
    with _tqdm(total=n_scenes, desc=os.path.basename(audio_dir)) as pbar:
        while scene_idx < n_scenes:
            n_src   = n_src_list[scene_idx % len(n_src_list)]
            sources = sample_sources_balanced(flat_segs, n_src, rng)
            if sources is None:
                skipped += 1
                if skipped > n_scenes * 5:
                    print(f'  WARNING: 소스 부족으로 {n_scenes - scene_idx}개 미생성')
                    break
                continue

            scene_id = base_id + scene_idx
            try:
                scene_data = build_scene(sources, scene_id, rng)
            except Exception as e:
                print(f'  씬 {scene_id} 생성 실패: {e}')
                skipped += 1
                continue

            save_scene(scene_data, audio_dir, anno_dir)
            scene_idx += 1
            pbar.update(1)

    return scene_idx


# tqdm 선택적 임포트
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _tqdm:
        def __init__(self, total=None, desc=''):
            self._n = 0; self._total = total; self._desc = desc
            print(f'{desc}: 0/{total}', end='\r')
        def update(self, n=1):
            self._n += n
            print(f'{self._desc}: {self._n}/{self._total}', end='\r')
        def __enter__(self): return self
        def __exit__(self, *a): print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MRS binaural 녹음을 azimuth-balanced 방식으로 혼합하여 데이터셋 생성',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mrs-root',      default='./MRSAudio/MRSLife/MRSSound')
    parser.add_argument('--out-dir',       default='./data_mrs_balanced')
    parser.add_argument('--n-train',       type=int, default=6000)
    parser.add_argument('--n-val',         type=int, default=750)
    parser.add_argument('--n-test',        type=int, default=250)
    parser.add_argument('--n-bins',        type=int, default=12,
                        help='Azimuth 균형 구간 수 (12=30°씩)')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--class-map-src', default='./data/meta/class_map.json')
    args = parser.parse_args()

    rng      = np.random.default_rng(args.seed)
    mrs_root = os.path.abspath(args.mrs_root)
    out_dir  = os.path.abspath(args.out_dir)

    # ── sound 디렉터리 분할 ────────────────────────────────────────────────────
    all_sounds  = sorted([
        d for d in os.listdir(mrs_root)
        if d.startswith('sound') and os.path.isdir(os.path.join(mrs_root, d))
    ])
    n_train_snd = int(len(all_sounds) * 0.8)
    train_sounds = all_sounds[:n_train_snd]
    test_sounds  = all_sounds[n_train_snd:]
    print(f'[POOL] train sounds: {len(train_sounds)}, test sounds: {len(test_sounds)}')

    # ── 세그먼트 풀 + 가중치 계산 ────────────────────────────────────────────
    print('\n[POOL] Train pool azimuth 분석:')
    flat_train = load_balanced_pool(mrs_root, train_sounds, n_bins=args.n_bins)
    print(f'       → {len(flat_train)} segs, '
          f'{len(set(s["snd_dir"] for s in flat_train))} sound dirs')

    print('\n[POOL] Test pool azimuth 분석:')
    flat_test = load_balanced_pool(mrs_root, test_sounds, n_bins=args.n_bins)
    print(f'       → {len(flat_test)} segs, '
          f'{len(set(s["snd_dir"] for s in flat_test))} sound dirs')

    # ── 출력 디렉터리 ──────────────────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(out_dir, 'audio',       split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'annotations', split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'meta'), exist_ok=True)

    base_val  = args.n_train
    base_test = args.n_train + args.n_val
    split_json = {
        'train': {'n_scenes': args.n_train, 'base_id': 0,
                  'audio_dir': os.path.join(out_dir, 'audio',       'train'),
                  'anno_dir' : os.path.join(out_dir, 'annotations', 'train')},
        'val':   {'n_scenes': args.n_val,   'base_id': base_val,
                  'audio_dir': os.path.join(out_dir, 'audio',       'val'),
                  'anno_dir' : os.path.join(out_dir, 'annotations', 'val')},
        'test':  {'n_scenes': args.n_test,  'base_id': base_test,
                  'audio_dir': os.path.join(out_dir, 'audio',       'test'),
                  'anno_dir' : os.path.join(out_dir, 'annotations', 'test')},
    }
    with open(os.path.join(out_dir, 'meta', 'split.json'), 'w') as f:
        json.dump(split_json, f, indent=2)

    cmap_src = os.path.abspath(args.class_map_src)
    cmap_dst = os.path.join(out_dir, 'meta', 'class_map.json')
    if os.path.exists(cmap_src):
        shutil.copy2(cmap_src, cmap_dst)
        print(f'\n[META] class_map.json 복사 완료')

    # ── 씬 생성 ────────────────────────────────────────────────────────────────
    print(f'\n[TRAIN] {args.n_train}개 씬 생성')
    n_done_train = build_split(
        flat_segs=flat_train, n_scenes=args.n_train, base_id=0,
        audio_dir=os.path.join(out_dir, 'audio',       'train'),
        anno_dir =os.path.join(out_dir, 'annotations', 'train'),
        rng=np.random.default_rng(args.seed),
    )

    print(f'\n[VAL]   {args.n_val}개 씬 생성')
    n_done_val = build_split(
        flat_segs=flat_train, n_scenes=args.n_val, base_id=base_val,
        audio_dir=os.path.join(out_dir, 'audio',       'val'),
        anno_dir =os.path.join(out_dir, 'annotations', 'val'),
        rng=np.random.default_rng(args.seed + 1),
    )

    print(f'\n[TEST]  {args.n_test}개 씬 생성')
    n_done_test = build_split(
        flat_segs=flat_test, n_scenes=args.n_test, base_id=base_test,
        audio_dir=os.path.join(out_dir, 'audio',       'test'),
        anno_dir =os.path.join(out_dir, 'annotations', 'test'),
        rng=np.random.default_rng(args.seed + 2),
        src_dist=(0.3, 0.4, 0.3),
    )

    split_json['train']['n_scenes'] = n_done_train
    split_json['val'  ]['n_scenes'] = n_done_val
    split_json['test' ]['n_scenes'] = n_done_test
    with open(os.path.join(out_dir, 'meta', 'split.json'), 'w') as f:
        json.dump(split_json, f, indent=2)

    print(f'\n[DONE]')
    print(f'  train: {n_done_train}  val: {n_done_val}  test: {n_done_test}')
    print(f'  출력: {out_dir}')


if __name__ == '__main__':
    main()
