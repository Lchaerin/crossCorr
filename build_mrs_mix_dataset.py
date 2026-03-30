#!/usr/bin/env python3
"""
build_mrs_mix_dataset.py
========================
MRS binaural 녹음들을 무작위로 혼합하여 multi-source 합성 데이터셋을 생성합니다.
출력 포맷은 기존 합성 데이터셋(data/)과 완전히 동일합니다.

출력 구조
---------
<out_dir>/
  audio/
    train/  scene_NNNNNN.wav + scene_NNNNNN.json
    val/
    test/
  annotations/
    train/  scene_NNNNNN_cls.npy  _doa.npy  _loud.npy  _mask.npy
    val/
    test/
  meta/
    split.json
    class_map.json   (기존 FSD50K class_map.json 복사)

설계 원칙
---------
- 한 씬(30초)에 1~3개 MRS 소스를 무작위 배치·혼합
- 소스마다 독립된 sound 디렉터리(→ 다른 악기) 에서 선택
- 각 소스는 씬 내 무작위 시작 시간에 배치
- DOA: MRS npy (right, fwd, up) → SLED 단위벡터 normalize([fwd, right, up])
- 클래스 ID: MRS_TO_FSD50K_MAP (기존 209-class 헤드와 호환)
- train 씬: MRS sound001–sound166 (80%) 에서 소스 선택
- val   씬: MRS sound001–sound166 (80%) 에서 소스 선택 (다른 조합)
- test  씬: MRS sound167–sound208 (20%) 에서 소스 선택 (held-out)

Usage
-----
    python build_mrs_mix_dataset.py \\
        --mrs-root ./MRSAudio/MRSLife/MRSSound \\
        --out-dir  ./data_mrs_mix \\
        --n-train  6000 \\
        --n-val    750  \\
        --n-test   250  \\
        --seed     42
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

# MRS → FSD50K 클래스 매핑 (기존 209-class 헤드와 호환)
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

# ── 상수 ────────────────────────────────────────────────────────────────────────
SR            = 48_000
HOP           = 960       # 20ms @ 48kHz
T_SCENE       = 1500      # 30s / 20ms
N_SAMPLES     = T_SCENE * HOP   # 1_440_000
MAX_SLOTS     = 3
MIN_LOUD_DB   = -60.0     # 이보다 조용한 프레임은 mask=False
TARGET_RMS    = 0.05      # 소스 정규화 목표 RMS (~-26 dBFS)
NPY_DT_MS     = 50.0      # MRS npy 어노테이션 간격 (ms)

# 소스 배치: 씬 내 최소/최대 시작 위치 (초)
SRC_START_MIN = 0.0
SRC_START_MAX = 20.0   # 최대 20s 시작 → 10s 소스가 30s 안에 끝남


# =============================================================================
# MRS 세그먼트 목록 로딩
# =============================================================================

def load_segment_pool(mrs_root: str, sound_dirs: list[str]) -> dict[str, list[dict]]:
    """sound 디렉터리별 세그먼트 목록 반환.

    Returns
    -------
    {snd_name: [seg_dict, ...]}
    각 seg_dict: wav_path, npy_path, class_id, start_ms, stop_ms
    """
    mrs_base = os.path.dirname(os.path.dirname(mrs_root))  # MRSAudio/
    pool: dict[str, list[dict]] = {}

    for snd in sound_dirs:
        snd_dir   = os.path.join(mrs_root, snd)
        meta_path = os.path.join(snd_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta_list = json.load(f)

        segs = []
        for seg in meta_list:
            event = seg.get('event', '')
            if event not in MRS_TO_FSD50K_MAP:
                continue
            class_id = MRS_TO_FSD50K_MAP[event]

            npy_path = os.path.join(mrs_base, seg['pos_fn'])
            wav_path = os.path.join(mrs_base, seg['wav_fn'])
            if not os.path.exists(npy_path) or not os.path.exists(wav_path):
                continue

            start_ms = seg['start'] * 1000.0
            stop_ms  = seg['stop']  * 1000.0
            dur_ms   = stop_ms - start_ms
            if dur_ms < 1000.0:   # 1초 미만 세그먼트 스킵
                continue

            segs.append({
                'wav_path' : wav_path,
                'npy_path' : npy_path,
                'class_id' : class_id,
                'event'    : event,
                'start_ms' : start_ms,
                'stop_ms'  : stop_ms,
            })

        if segs:
            pool[snd] = segs

    return pool


# =============================================================================
# 단일 소스 오디오 + DOA 로딩
# =============================================================================

def load_source(seg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """세그먼트 오디오와 DOA 배열 로드.

    Returns
    -------
    audio   : [2, N_seg]  float32  (정규화 전)
    doa_seq : [T_seg, 3]  float32  SLED 단위벡터 (20ms 프레임마다)
    times_ms: [T_seg]     float64  각 프레임 중심 시각 (절대 ms)
    """
    start_ms = seg['start_ms']
    stop_ms  = seg['stop_ms']
    dur_ms   = stop_ms - start_ms

    # ── 오디오 로드 ────────────────────────────────────────────────────────────
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

    # ── npy DOA 로드 & 보간 ────────────────────────────────────────────────────
    npy = np.load(seg['npy_path'])  # [T_npy, 4]: (right, fwd, up, time_ms)
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

    # MRS (right, forward, up) → SLED (x=fwd, y=right, z=up) → 정규화
    doa_raw = np.stack([forward_f, right_f, up_f], axis=1)  # [T_seg, 3]
    norm    = np.linalg.norm(doa_raw, axis=1, keepdims=True)
    norm    = np.where(norm < 1e-8, 1.0, norm)
    doa_seq = (doa_raw / norm).astype(np.float32)           # [T_seg, 3]

    return audio_np, doa_seq, frame_times_ms


# =============================================================================
# 씬 생성
# =============================================================================

def build_scene(sources: list[dict], scene_id: int, rng: np.random.Generator) -> dict:
    """소스 목록으로 30초 씬 생성.

    Parameters
    ----------
    sources : list of seg dicts (1–3개, 서로 다른 sound 디렉터리)
    scene_id: 씬 번호
    rng     : numpy random generator

    Returns
    -------
    {audio, cls, doa, loud, mask, meta_json}
    """
    n_src = len(sources)
    audio_mix = np.zeros((2, N_SAMPLES), dtype=np.float32)

    cls_arr  = np.full((T_SCENE, MAX_SLOTS), -1,    dtype=np.int16)
    doa_arr  = np.zeros((T_SCENE, MAX_SLOTS, 3),    dtype=np.float32)
    loud_arr = np.full((T_SCENE, MAX_SLOTS), -80.0, dtype=np.float32)
    mask_arr = np.zeros((T_SCENE, MAX_SLOTS),       dtype=bool)

    events_json = []

    for slot, seg in enumerate(sources):
        audio_src, doa_seq, _ = load_source(seg)
        T_seg   = doa_seq.shape[0]
        N_seg   = audio_src.shape[1]

        # ── 소스 정규화 → target RMS ──────────────────────────────────────────
        rms = float(np.sqrt(np.mean(audio_src ** 2)))
        if rms > 1e-8:
            audio_src = audio_src * (TARGET_RMS / rms)
        else:
            audio_src = audio_src * TARGET_RMS

        # ── 랜덤 게인 (±6 dB) ────────────────────────────────────────────────
        gain_db  = float(rng.uniform(-6.0, 6.0))
        gain     = 10.0 ** (gain_db / 20.0)
        audio_src = audio_src * gain

        # ── 씬 내 랜덤 시작 위치 (소스 길이가 씬 안에 들어오도록) ─────────────
        max_start_s  = max(0.0, SRC_START_MAX - (T_seg * 0.02))
        scene_start_s = float(rng.uniform(SRC_START_MIN, max(SRC_START_MIN, max_start_s)))
        scene_start_smp = int(scene_start_s * SR)
        scene_end_smp   = min(scene_start_smp + N_seg, N_SAMPLES)
        actual_len_smp  = scene_end_smp - scene_start_smp

        # ── 오디오 믹스에 추가 ─────────────────────────────────────────────────
        audio_mix[:, scene_start_smp:scene_end_smp] += audio_src[:, :actual_len_smp]

        # ── 어노테이션 채우기 ─────────────────────────────────────────────────
        scene_start_frame = scene_start_smp // HOP
        actual_len_frames = min(T_seg, (actual_len_smp + HOP - 1) // HOP)
        scene_end_frame   = min(scene_start_frame + actual_len_frames, T_SCENE)

        for t_scene in range(scene_start_frame, scene_end_frame):
            t_local = t_scene - scene_start_frame
            if t_local >= T_seg:
                break

            cls_arr[t_scene, slot]  = seg['class_id']
            doa_arr[t_scene, slot]  = doa_seq[t_local]

            # 해당 프레임 소스 단독 RMS → loudness
            s0 = t_local * HOP
            s1 = min(s0 + HOP, audio_src.shape[1])
            if s1 > s0:
                frame_rms = float(np.sqrt(np.mean(audio_src[:, s0:s1] ** 2)))
            else:
                frame_rms = 0.0
            loud_db = 20.0 * np.log10(frame_rms + 1e-9)
            loud_arr[t_scene, slot] = float(np.clip(loud_db, -80.0, 0.0))
            mask_arr[t_scene, slot] = loud_db > MIN_LOUD_DB

        # ── JSON 이벤트 기록 ──────────────────────────────────────────────────
        events_json.append({
            'file'        : f"MRS/{seg['event']}",
            'start_sample': int(scene_start_smp),
            'end_sample'  : int(scene_end_smp),
            'start_time'  : round(scene_start_smp / SR, 4),
            'end_time'    : round(scene_end_smp   / SR, 4),
            'azimuth'     : None,   # DOA npy에서 시간별로 변함
            'elevation'   : None,
            'gain'        : round(float(gain), 4),
            'source_event': seg['event'],
            'class_id'    : int(seg['class_id']),
        })

    # ── 믹스 클리핑 방지 ──────────────────────────────────────────────────────
    peak = float(np.abs(audio_mix).max())
    if peak > 0.95:
        audio_mix = audio_mix * (0.95 / peak)
        # loud_arr도 비례 보정
        scale_db = 20.0 * np.log10(0.95 / peak)
        loud_arr = np.clip(loud_arr + scale_db, -80.0, 0.0)
        # mask 재계산
        mask_arr = loud_arr > MIN_LOUD_DB
        # 비활성 슬롯 보정 (cls=-1인 곳은 다시 -80으로)
        for sl in range(MAX_SLOTS):
            inactive = cls_arr[:, sl] == -1
            loud_arr[inactive, sl] = -80.0
            mask_arr[inactive, sl] = False

    scene_name = f'scene_{scene_id:06d}'
    meta_json = {
        'scene_name' : scene_name,
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
        'audio' : audio_mix,          # [2, N_SAMPLES] float32
        'cls'   : cls_arr,            # [T_SCENE, 3] int16
        'doa'   : doa_arr,            # [T_SCENE, 3, 3] float32
        'loud'  : loud_arr,           # [T_SCENE, 3] float32
        'mask'  : mask_arr,           # [T_SCENE, 3] bool
        'meta'  : meta_json,
    }


# =============================================================================
# 씬 저장
# =============================================================================

def save_scene(scene_data: dict, audio_dir: str, anno_dir: str):
    name = scene_data['meta']['scene_name']

    # WAV
    wav_path = os.path.join(audio_dir, f'{name}.wav')
    audio_T  = scene_data['audio'].T.astype(np.float32)  # [N, 2]
    sf.write(wav_path, audio_T, SR, subtype='FLOAT')

    # JSON
    json_path = os.path.join(audio_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(scene_data['meta'], f, indent=2)

    # npy annotations
    np.save(os.path.join(anno_dir, f'{name}_cls.npy'),
            scene_data['cls'].astype(np.int16))
    np.save(os.path.join(anno_dir, f'{name}_doa.npy'),
            scene_data['doa'].astype(np.float16))
    np.save(os.path.join(anno_dir, f'{name}_loud.npy'),
            scene_data['loud'].astype(np.float16))
    np.save(os.path.join(anno_dir, f'{name}_mask.npy'),
            scene_data['mask'])


# =============================================================================
# 소스 조합 선택
# =============================================================================

def sample_sources(pool: dict[str, list[dict]], n_src: int,
                   rng: np.random.Generator) -> list[dict] | None:
    """pool에서 n_src개 소스를 서로 다른 sound 디렉터리에서 선택."""
    if len(pool) < n_src:
        return None
    chosen_dirs = rng.choice(list(pool.keys()), size=n_src, replace=False)
    sources = []
    for snd_dir in chosen_dirs:
        segs = pool[snd_dir]
        sources.append(segs[int(rng.integers(len(segs)))])
    return sources


# =============================================================================
# 스플릿 생성
# =============================================================================

def build_split(pool: dict[str, list[dict]],
                n_scenes: int, base_id: int,
                audio_dir: str, anno_dir: str,
                rng: np.random.Generator,
                src_dist: tuple[float, float, float] = (0.2, 0.4, 0.4)):
    """n_scenes개 씬을 생성하고 저장."""
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(anno_dir,  exist_ok=True)

    # 소스 수 분포: 1, 2, 3
    n_per_src = [
        int(n_scenes * src_dist[0]),
        int(n_scenes * src_dist[1]),
        n_scenes - int(n_scenes * src_dist[0]) - int(n_scenes * src_dist[1]),
    ]
    n_src_list = ([1] * n_per_src[0] + [2] * n_per_src[1] + [3] * n_per_src[2])
    rng.shuffle(n_src_list)

    skipped = 0
    scene_idx = 0
    with _tqdm(total=n_scenes, desc=os.path.basename(audio_dir)) as pbar:
        while scene_idx < n_scenes:
            n_src   = n_src_list[scene_idx % len(n_src_list)]
            sources = sample_sources(pool, n_src, rng)
            if sources is None:
                skipped += 1
                if skipped > n_scenes * 3:
                    print(f'  WARNING: pool 부족으로 {n_scenes - scene_idx}개 미생성')
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
    import contextlib
    class _tqdm:
        def __init__(self, total=None, desc=''):
            self._n = 0; self._total = total; self._desc = desc
            print(f'{desc}: 0/{total}', end='\r')
        def update(self, n=1):
            self._n += n
            print(f'{self._desc}: {self._n}/{self._total}', end='\r')
        def __enter__(self): return self
        def __exit__(self, *a):
            print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MRS 비나우럴 녹음을 혼합하여 multi-source 학습 데이터셋 생성',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mrs-root', default='./MRSAudio/MRSLife/MRSSound',
                        help='MRSSound 디렉터리 경로')
    parser.add_argument('--out-dir',  default='./data_mrs_mix',
                        help='출력 데이터셋 루트 디렉터리')
    parser.add_argument('--n-train',  type=int, default=6000)
    parser.add_argument('--n-val',    type=int, default=750)
    parser.add_argument('--n-test',   type=int, default=250)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--class-map-src',
                        default='./data/meta/class_map.json',
                        help='복사할 기존 class_map.json 경로')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    mrs_root = os.path.abspath(args.mrs_root)
    out_dir  = os.path.abspath(args.out_dir)

    # ── sound 디렉터리 목록 ───────────────────────────────────────────────────
    all_sounds = sorted([
        d for d in os.listdir(mrs_root)
        if d.startswith('sound') and os.path.isdir(os.path.join(mrs_root, d))
    ])
    n_total  = len(all_sounds)
    n_train_snd = int(n_total * 0.8)
    train_sounds = all_sounds[:n_train_snd]   # sound001–sound166
    test_sounds  = all_sounds[n_train_snd:]   # sound167–sound208

    print(f'[POOL] train sounds: {len(train_sounds)}, test sounds: {len(test_sounds)}')

    pool_train = load_segment_pool(mrs_root, train_sounds)
    pool_test  = load_segment_pool(mrs_root, test_sounds)
    print(f'       train pool: {len(pool_train)} dirs  '
          f'({sum(len(v) for v in pool_train.values())} segs)')
    print(f'       test  pool: {len(pool_test)} dirs  '
          f'({sum(len(v) for v in pool_test.values())} segs)')

    # ── 출력 디렉터리 ──────────────────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(out_dir, 'audio',       split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'annotations', split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'meta'), exist_ok=True)

    # ── split.json 작성 ───────────────────────────────────────────────────────
    base_val  = args.n_train
    base_test = args.n_train + args.n_val
    split_json = {
        'train': {
            'n_scenes' : args.n_train,
            'base_id'  : 0,
            'audio_dir': os.path.join(out_dir, 'audio',       'train'),
            'anno_dir' : os.path.join(out_dir, 'annotations', 'train'),
        },
        'val': {
            'n_scenes' : args.n_val,
            'base_id'  : base_val,
            'audio_dir': os.path.join(out_dir, 'audio',       'val'),
            'anno_dir' : os.path.join(out_dir, 'annotations', 'val'),
        },
        'test': {
            'n_scenes' : args.n_test,
            'base_id'  : base_test,
            'audio_dir': os.path.join(out_dir, 'audio',       'test'),
            'anno_dir' : os.path.join(out_dir, 'annotations', 'test'),
        },
    }
    with open(os.path.join(out_dir, 'meta', 'split.json'), 'w') as f:
        json.dump(split_json, f, indent=2)

    # ── class_map.json 복사 (기존 FSD50K 맵 재사용) ───────────────────────────
    cmap_src = os.path.abspath(args.class_map_src)
    cmap_dst = os.path.join(out_dir, 'meta', 'class_map.json')
    if os.path.exists(cmap_src):
        shutil.copy2(cmap_src, cmap_dst)
        print(f'[META] class_map.json 복사: {cmap_src} → {cmap_dst}')
    else:
        print(f'[META] class_map_src 없음 ({cmap_src}), 스킵')

    # ── 씬 생성 ────────────────────────────────────────────────────────────────
    print(f'\n[TRAIN] {args.n_train}개 씬 생성 (pool: train sounds)')
    n_done_train = build_split(
        pool=pool_train,
        n_scenes=args.n_train,
        base_id=0,
        audio_dir=os.path.join(out_dir, 'audio',       'train'),
        anno_dir =os.path.join(out_dir, 'annotations', 'train'),
        rng=np.random.default_rng(args.seed),
    )

    print(f'\n[VAL]   {args.n_val}개 씬 생성 (pool: train sounds, 다른 시드)')
    n_done_val = build_split(
        pool=pool_train,
        n_scenes=args.n_val,
        base_id=base_val,
        audio_dir=os.path.join(out_dir, 'audio',       'val'),
        anno_dir =os.path.join(out_dir, 'annotations', 'val'),
        rng=np.random.default_rng(args.seed + 1),
    )

    print(f'\n[TEST]  {args.n_test}개 씬 생성 (pool: held-out test sounds)')
    n_done_test = build_split(
        pool=pool_test,
        n_scenes=args.n_test,
        base_id=base_test,
        audio_dir=os.path.join(out_dir, 'audio',       'test'),
        anno_dir =os.path.join(out_dir, 'annotations', 'test'),
        rng=np.random.default_rng(args.seed + 2),
        src_dist=(0.3, 0.4, 0.3),   # test는 더 균형있게
    )

    # ── split.json 실제 생성 수로 업데이트 ────────────────────────────────────
    split_json['train']['n_scenes'] = n_done_train
    split_json['val'  ]['n_scenes'] = n_done_val
    split_json['test' ]['n_scenes'] = n_done_test
    with open(os.path.join(out_dir, 'meta', 'split.json'), 'w') as f:
        json.dump(split_json, f, indent=2)

    print(f'\n[DONE]')
    print(f'  train: {n_done_train}개 씬 → {out_dir}/audio/train/')
    print(f'  val  : {n_done_val}개 씬 → {out_dir}/audio/val/')
    print(f'  test : {n_done_test}개 씬 → {out_dir}/annotations/test/')
    print(f'  split.json → {out_dir}/meta/split.json')


if __name__ == '__main__':
    main()
