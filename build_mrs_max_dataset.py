#!/usr/bin/env python3
"""
build_mrs_max_dataset.py
========================
MRSSound + MRSSpeech + MRSDialogue 세 소스를 합쳐서
azimuth × elevation 2D 균형 샘플링으로 multi-source 데이터셋을 생성합니다.

동기
----
- 기존 MRSSound 단독 데이터셋은 후면(azimuth ≈ ±180°) 및
  상하(elevation ≠ 0°) 소스가 부족 → 후면 인식률 저하
- MRSSpeech / MRSDialogue 추가로 소스 다양성 확보
- 2D (az, el) 구간별 가중치 → 모든 방향이 균등 샘플링됨

클래스 매핑
-----------
  MRSSound  : 기존 MRS_TO_FSD50K_MAP 그대로 (29개 이벤트)
  MRSSpeech : Speech (157) – 드라마 대사
  MRSDialogue: Conversation (43) – 일상 대화

출력 포맷
---------
  <out_dir>/
    audio/        train|val|test/  scene_NNNNNN.wav  .json
    annotations/  train|val|test/  scene_NNNNNN_{cls,doa,loud,mask}.npy
    meta/         split.json  class_map.json

Usage
-----
    python build_mrs_max_dataset.py \\
        --sound-root    ./MRSAudio/MRSLife/MRSSound \\
        --speech-root   ./MRSAudio/MRSSpeech/MRSSpeech \\
        --dialogue-root ./MRSAudio/MRSDialogue/MRSLife/MRSDialogue \\
        --out-dir       ./data_mrs_max \\
        --n-train 8000  --n-val 1000  --n-test 500 \\
        --n-az-bins 12  --n-el-bins 5 \\
        --seed 42
"""

import argparse
import json
import os
import shutil

import numpy as np
import soundfile as sf

# ── 클래스 매핑 ───────────────────────────────────────────────────────────────
MRS_SOUND_MAP: dict[str, int] = {
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
CLASS_SPEECH    = 157   # Speech
CLASS_DIALOGUE  =  43   # Conversation

# ── 오디오 상수 ───────────────────────────────────────────────────────────────
SR            = 48_000
HOP           = 960        # 20ms @ 48kHz
T_SCENE       = 1500       # 30s / 20ms
N_SAMPLES     = T_SCENE * HOP
MAX_SLOTS     = 3
MIN_LOUD_DB   = -60.0
TARGET_RMS    = 0.05
SRC_START_MIN = 0.0
SRC_START_MAX = 20.0


# =============================================================================
# DOA 유틸
# =============================================================================

def _mean_az_el(npy_path: str) -> tuple[float, float]:
    """npy 에서 circular-mean azimuth(°), elevation(°) 계산.

    npy 컬럼: [right_m, fwd_m, up_m, time_ms, ...]
    az = atan2(right, fwd)  (0°=정면, 90°=오른쪽)
    el = atan2(up, sqrt(right²+fwd²))
    """
    npy   = np.atleast_2d(np.load(npy_path))   # (5,) 단일행 npz 대응
    right = npy[:, 0]
    fwd   = npy[:, 1]
    up    = npy[:, 2]

    az_rad = np.arctan2(right, fwd)
    el_rad = np.arctan2(up, np.sqrt(right ** 2 + fwd ** 2))

    mean_az = np.arctan2(np.mean(np.sin(az_rad)), np.mean(np.cos(az_rad)))
    mean_el = np.arctan2(np.mean(np.sin(el_rad)), np.mean(np.cos(el_rad)))
    return float(np.degrees(mean_az)), float(np.degrees(mean_el))


def _assign_bins(az_deg: float, el_deg: float,
                 az_edges: np.ndarray, el_edges: np.ndarray) -> int:
    n_az = len(az_edges) - 1
    n_el = len(el_edges) - 1
    az_b = int(np.clip(np.digitize(az_deg, az_edges) - 1, 0, n_az - 1))
    el_b = int(np.clip(np.digitize(el_deg, el_edges) - 1, 0, n_el - 1))
    return az_b * n_el + el_b


# =============================================================================
# 세그먼트 풀 로딩
# =============================================================================

def _load_sound_pool(sound_root: str, dirs: list[str]) -> list[dict]:
    """MRSSound 세그먼트 풀 로딩.

    npy: (T, 4)  [right, fwd, up, time_ms]
    wav: 공유 binaural wav, start/stop 슬라이스 필요
    """
    mrs_base = os.path.dirname(os.path.dirname(sound_root))  # ./MRSAudio/
    segs: list[dict] = []

    for snd in dirs:
        snd_dir   = os.path.join(sound_root, snd)
        meta_path = os.path.join(snd_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta_list = json.load(f)

        for seg in meta_list:
            event = seg.get('event', '')
            if event not in MRS_SOUND_MAP:
                continue

            npy_path = os.path.join(mrs_base, seg['pos_fn'])
            wav_path = os.path.join(mrs_base, seg['wav_fn'])
            if not os.path.exists(npy_path) or not os.path.exists(wav_path):
                continue

            start_ms = seg['start'] * 1000.0
            stop_ms  = seg['stop']  * 1000.0
            if stop_ms - start_ms < 1000.0:
                continue

            segs.append({
                'wav_path'      : wav_path,
                'npy_path'      : npy_path,
                'class_id'      : MRS_SOUND_MAP[event],
                'event'         : event,
                'start_ms'      : start_ms,
                'stop_ms'       : stop_ms,
                'wav_offset_ms' : start_ms,   # 공유 wav에서 슬라이스
                'src_key'       : f'sound/{snd}',
            })

    return segs


def _load_speech_pool(speech_root: str, dirs: list[str]) -> list[dict]:
    """MRSSpeech 세그먼트 풀 로딩.

    npy: (T, 5)  [right, fwd, up, time_ms, ...]
    wav: 세그먼트별 개별 wav, start/stop 슬라이스
    """
    speech_base = os.path.dirname(speech_root)   # ./MRSAudio/MRSSpeech
    segs: list[dict] = []

    for drama in dirs:
        drama_dir  = os.path.join(speech_root, drama)
        meta_path  = os.path.join(drama_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta_list = json.load(f)

        for seg in meta_list:
            npy_path = os.path.join(speech_base, seg['pos_fn'])
            wav_path = os.path.join(speech_base, seg['wav_fn'])
            if not os.path.exists(npy_path) or not os.path.exists(wav_path):
                continue

            start_ms = seg.get('start', 0.0) * 1000.0
            stop_ms  = seg.get('stop',  0.0) * 1000.0

            # start/stop 없으면 npy 시간 범위 사용
            if stop_ms <= start_ms:
                npy = np.load(npy_path)
                start_ms = float(npy[0,  3])
                stop_ms  = float(npy[-1, 3])

            if stop_ms - start_ms < 1000.0:
                continue

            segs.append({
                'wav_path'      : wav_path,
                'npy_path'      : npy_path,
                'class_id'      : CLASS_SPEECH,
                'event'         : 'speech',
                'start_ms'      : start_ms,
                'stop_ms'       : stop_ms,
                'wav_offset_ms' : start_ms,   # 세그먼트 wav 내 슬라이스
                'src_key'       : f'speech/{drama}',
            })

    return segs


def _load_dialogue_pool(dialogue_root: str, dirs: list[str]) -> list[dict]:
    """MRSDialogue 세그먼트 풀 로딩.

    npy: (T, 5)  [right, fwd, up, time_ms, ...]
    wav: 세그먼트별 개별 wav, 전체 파일 사용
    metadata에 start/stop 없음 → npy 시간 범위로 대체
    """
    dialogue_base = os.path.dirname(os.path.dirname(dialogue_root))  # ./MRSAudio/MRSDialogue
    segs: list[dict] = []

    for dlg in dirs:
        dlg_dir   = os.path.join(dialogue_root, dlg)
        meta_path = os.path.join(dlg_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta_list = json.load(f)

        for seg in meta_list:
            npy_path = os.path.join(dialogue_base, seg['pos_fn'])
            wav_path = os.path.join(dialogue_base, seg['wav_fn'])
            if not os.path.exists(npy_path) or not os.path.exists(wav_path):
                continue

            # MRSDialogue는 start/stop 없음 → npy 시간 사용
            npy      = np.load(npy_path)
            start_ms = float(npy[0,  3])
            stop_ms  = float(npy[-1, 3])

            if stop_ms - start_ms < 500.0:   # MRSDialogue는 짧은 발화 허용(0.5s)
                continue

            segs.append({
                'wav_path'      : wav_path,
                'npy_path'      : npy_path,
                'class_id'      : CLASS_DIALOGUE,
                'event'         : 'dialogue',
                'start_ms'      : start_ms,
                'stop_ms'       : stop_ms,
                'wav_offset_ms' : 0.0,    # 개별 wav 전체 사용 (scene-time 오프셋 없음)
                'src_key'       : f'dialogue/{dlg}',
            })

    return segs


# =============================================================================
# 2D 균형 풀 구성
# =============================================================================

def build_balanced_pool(segs: list[dict],
                        n_az: int = 12,
                        n_el: int = 5) -> list[dict]:
    """모든 세그먼트에 (az, el) bin 및 weight 부여.

    weight = 1 / (해당 2D bin 내 세그먼트 수)
    → 모든 (az, el) bin이 동일 확률로 선택됨
    """
    az_edges = np.linspace(-180, 180, n_az + 1)
    el_edges = np.linspace(-90,   90, n_el + 1)
    n_bins   = n_az * n_el

    print(f'  DOA 분석 중 ({len(segs)} segs)...')
    for s in segs:
        az, el = _mean_az_el(s['npy_path'])
        b      = _assign_bins(az, el, az_edges, el_edges)
        s['mean_az']  = az
        s['mean_el']  = el
        s['doa_bin']  = b
        s['weight']   = 0.0

    bin_counts = np.zeros(n_bins, dtype=int)
    for s in segs:
        bin_counts[s['doa_bin']] += 1

    # 분포 출력
    print(f'  Azimuth × Elevation 분포 ({n_az}az × {n_el}el = {n_bins} bins):')
    for az_i in range(n_az):
        az_lo = az_edges[az_i]; az_hi = az_edges[az_i + 1]
        row_counts = []
        for el_i in range(n_el):
            el_lo = el_edges[el_i]; el_hi = el_edges[el_i + 1]
            b = az_i * n_el + el_i
            row_counts.append(f'el[{el_lo:.0f}~{el_hi:.0f}]:{bin_counts[b]:4d}')
        print(f'    az[{az_lo:5.0f}~{az_hi:5.0f}]: ' + '  '.join(row_counts))

    for s in segs:
        cnt = bin_counts[s['doa_bin']]
        s['weight'] = 1.0 / cnt if cnt > 0 else 0.0

    return segs


# =============================================================================
# 소스 로딩 (통합)
# =============================================================================

def load_source(seg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """세그먼트 오디오 + DOA 로드 (MRSSound / MRSSpeech / MRSDialogue 공통).

    Parameters
    ----------
    seg : dict  (wav_path, npy_path, start_ms, stop_ms, wav_offset_ms)

    Returns
    -------
    audio   : [2, N_seg]  float32
    doa_seq : [T_seg, 3]  float32  SLED 단위벡터 (fwd, right, up) @20ms
    frame_times_ms : [T_seg] float64
    """
    start_ms    = seg['start_ms']
    stop_ms     = seg['stop_ms']
    dur_ms      = stop_ms - start_ms
    wav_off_ms  = seg.get('wav_offset_ms', start_ms)

    # ── 오디오 로드 ────────────────────────────────────────────────────────────
    s_start = int(wav_off_ms * SR / 1000)
    s_stop  = int((wav_off_ms + dur_ms) * SR / 1000)
    try:
        audio_np, _ = sf.read(
            seg['wav_path'], start=s_start, stop=s_stop,
            dtype='float32', always_2d=True,
        )
    except Exception:
        n_exp    = int(dur_ms * SR / 1000)
        audio_np = np.zeros((n_exp, 2), dtype=np.float32)

    audio_np = audio_np.T  # [2, N_seg]

    # ── npy DOA 로드 + 보간 ────────────────────────────────────────────────────
    npy         = np.atleast_2d(np.load(seg['npy_path']))  # (T, 4+); 1D 대응
    npy_times   = npy[:, 3]
    npy_right   = npy[:, 0]
    npy_forward = npy[:, 1]
    npy_up      = npy[:, 2]

    T_seg          = max(1, int(dur_ms / 20.0))
    frame_times_ms = start_ms + (np.arange(T_seg) + 0.5) * 20.0
    frame_times_cl = np.clip(frame_times_ms, npy_times[0], npy_times[-1])

    right_f   = np.interp(frame_times_cl, npy_times, npy_right)
    forward_f = np.interp(frame_times_cl, npy_times, npy_forward)
    up_f      = np.interp(frame_times_cl, npy_times, npy_up)

    # MRS (right, fwd, up) → SLED (x=fwd, y=right, z=up) → 정규화
    doa_raw = np.stack([forward_f, right_f, up_f], axis=1)  # [T, 3]
    norm    = np.linalg.norm(doa_raw, axis=1, keepdims=True)
    norm    = np.where(norm < 1e-8, 1.0, norm)
    doa_seq = (doa_raw / norm).astype(np.float32)

    return audio_np, doa_seq, frame_times_ms


# =============================================================================
# 균형 샘플링
# =============================================================================

def sample_balanced(flat_segs: list[dict], n_src: int,
                    rng: np.random.Generator) -> list[dict] | None:
    """2D bin 가중치 기반 샘플링, 동일 src_key 중복 불허."""
    weights = np.array([s['weight'] for s in flat_segs], dtype=np.float64)
    mask    = np.ones(len(flat_segs), dtype=bool)
    chosen  : list[dict] = []
    used    : set[str]   = set()

    for _ in range(n_src):
        w = weights * mask
        total_w = w.sum()
        if total_w < 1e-12:
            return None
        p   = w / total_w
        idx = int(rng.choice(len(flat_segs), p=p))

        seg = flat_segs[idx]
        chosen.append(seg)
        used.add(seg['src_key'])

        # 같은 src_key 마스킹
        for i, s in enumerate(flat_segs):
            if s['src_key'] == seg['src_key']:
                mask[i] = False

    return chosen if len(chosen) == n_src else None


# =============================================================================
# 씬 생성 · 저장
# =============================================================================

def build_scene(sources: list[dict], scene_id: int,
                rng: np.random.Generator) -> dict:
    n_src     = len(sources)
    audio_mix = np.zeros((2, N_SAMPLES), dtype=np.float32)
    cls_arr   = np.full((T_SCENE, MAX_SLOTS), -1,    dtype=np.int16)
    doa_arr   = np.zeros((T_SCENE, MAX_SLOTS, 3),    dtype=np.float32)
    loud_arr  = np.full((T_SCENE, MAX_SLOTS), -80.0, dtype=np.float32)
    mask_arr  = np.zeros((T_SCENE, MAX_SLOTS),       dtype=bool)
    events_json: list[dict] = []

    for slot, seg in enumerate(sources):
        audio_src, doa_seq, _ = load_source(seg)
        T_seg = doa_seq.shape[0]
        N_seg = audio_src.shape[1]

        # 정규화 → target RMS
        rms = float(np.sqrt(np.mean(audio_src ** 2)))
        audio_src = audio_src * (TARGET_RMS / rms) if rms > 1e-8 else audio_src * TARGET_RMS

        # ±6 dB 랜덤 게인
        gain    = 10.0 ** (float(rng.uniform(-6.0, 6.0)) / 20.0)
        audio_src *= gain

        # 씬 내 랜덤 시작 위치
        max_start_s     = max(0.0, SRC_START_MAX - T_seg * 0.02)
        scene_start_s   = float(rng.uniform(SRC_START_MIN, max(SRC_START_MIN, max_start_s)))
        scene_start_smp = int(scene_start_s * SR)
        scene_end_smp   = min(scene_start_smp + N_seg, N_SAMPLES)
        actual_smp      = scene_end_smp - scene_start_smp

        audio_mix[:, scene_start_smp:scene_end_smp] += audio_src[:, :actual_smp]

        f_start = scene_start_smp // HOP
        f_len   = min(T_seg, (actual_smp + HOP - 1) // HOP)
        f_end   = min(f_start + f_len, T_SCENE)

        for t in range(f_start, f_end):
            tl = t - f_start
            if tl >= T_seg:
                break
            cls_arr[t, slot] = seg['class_id']
            doa_arr[t, slot] = doa_seq[tl]
            s0 = tl * HOP
            s1 = min(s0 + HOP, audio_src.shape[1])
            frms = float(np.sqrt(np.mean(audio_src[:, s0:s1] ** 2))) if s1 > s0 else 0.0
            loud_db = float(np.clip(20.0 * np.log10(frms + 1e-9), -80.0, 0.0))
            loud_arr[t, slot] = loud_db
            mask_arr[t, slot] = loud_db > MIN_LOUD_DB

        events_json.append({
            'file'        : f"MRS/{seg['event']}",
            'start_time'  : round(scene_start_smp / SR, 4),
            'end_time'    : round(scene_end_smp   / SR, 4),
            'start_sample': int(scene_start_smp),
            'end_sample'  : int(scene_end_smp),
            'gain'        : round(float(gain), 4),
            'source_event': seg['event'],
            'class_id'    : int(seg['class_id']),
            'mean_az_deg' : round(seg.get('mean_az', 0.0), 2),
            'mean_el_deg' : round(seg.get('mean_el', 0.0), 2),
            'src_key'     : seg['src_key'],
        })

    # 클리핑 방지
    peak = float(np.abs(audio_mix).max())
    if peak > 0.95:
        audio_mix *= 0.95 / peak
        scale_db  = 20.0 * np.log10(0.95 / peak)
        loud_arr  = np.clip(loud_arr + scale_db, -80.0, 0.0)
        mask_arr  = loud_arr > MIN_LOUD_DB
        for sl in range(MAX_SLOTS):
            inactive = cls_arr[:, sl] == -1
            loud_arr[inactive, sl] = -80.0
            mask_arr[inactive, sl] = False

    scene_name = f'scene_{scene_id:06d}'
    return {
        'audio': audio_mix,
        'cls'  : cls_arr,
        'doa'  : doa_arr,
        'loud' : loud_arr,
        'mask' : mask_arr,
        'meta' : {
            'scene_name'  : scene_name,
            'duration_sec': round(T_SCENE * HOP / SR, 4),
            'sample_rate' : SR,
            'hop_samples' : HOP,
            'n_frames'    : T_SCENE,
            'max_slots'   : MAX_SLOTS,
            'audio_file'  : f'{scene_name}.wav',
            'num_events'  : n_src,
            'events'      : events_json,
        },
    }


def save_scene(data: dict, audio_dir: str, anno_dir: str):
    name = data['meta']['scene_name']
    sf.write(os.path.join(audio_dir, f'{name}.wav'),
             data['audio'].T.astype(np.float32), SR, subtype='FLOAT')
    with open(os.path.join(audio_dir, f'{name}.json'), 'w') as f:
        json.dump(data['meta'], f, indent=2)
    np.save(os.path.join(anno_dir, f'{name}_cls.npy'),  data['cls'].astype(np.int16))
    np.save(os.path.join(anno_dir, f'{name}_doa.npy'),  data['doa'].astype(np.float16))
    np.save(os.path.join(anno_dir, f'{name}_loud.npy'), data['loud'].astype(np.float16))
    np.save(os.path.join(anno_dir, f'{name}_mask.npy'), data['mask'])


# =============================================================================
# 스플릿 생성
# =============================================================================

def build_split(flat_segs: list[dict],
                n_scenes: int, base_id: int,
                audio_dir: str, anno_dir: str,
                rng: np.random.Generator,
                src_dist: tuple[float, float, float] = (0.2, 0.4, 0.4)) -> int:
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(anno_dir,  exist_ok=True)

    n_dirs = len(set(s['src_key'] for s in flat_segs))
    p = src_dist
    n1 = int(n_scenes * p[0])
    n2 = int(n_scenes * p[1])
    n3 = n_scenes - n1 - n2
    n_src_list = [1]*n1 + [2]*n2 + [3]*n3
    rng.shuffle(n_src_list)

    if n_dirs < 3:
        print(f'  WARNING: src_key {n_dirs}개뿐, n_src 최대 {n_dirs}로 제한')
        n_src_list = [min(n, n_dirs) for n in n_src_list]

    skipped   = 0
    scene_idx = 0
    with _tqdm(total=n_scenes, desc=os.path.basename(audio_dir)) as pbar:
        while scene_idx < n_scenes:
            n_src   = n_src_list[scene_idx % len(n_src_list)]
            sources = sample_balanced(flat_segs, n_src, rng)
            if sources is None:
                skipped += 1
                if skipped > n_scenes * 5:
                    print(f'  WARNING: 소스 부족으로 {n_scenes - scene_idx}개 미생성')
                    break
                continue

            try:
                scene_data = build_scene(sources, base_id + scene_idx, rng)
            except Exception as e:
                print(f'  씬 {base_id + scene_idx} 실패: {e}')
                skipped += 1
                continue

            save_scene(scene_data, audio_dir, anno_dir)
            scene_idx += 1
            pbar.update(1)

    return scene_idx


# tqdm fallback
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _tqdm:
        def __init__(self, total=None, desc=''):
            self._n = 0; self._t = total; self._d = desc
            print(f'{desc}: 0/{total}', end='\r')
        def update(self, n=1):
            self._n += n
            print(f'{self._d}: {self._n}/{self._t}', end='\r')
        def __enter__(self): return self
        def __exit__(self, *a): print()


# =============================================================================
# 헬퍼: 소스 디렉터리 80/20 분할
# =============================================================================

def _split_dirs(root: str, prefix: str = '') -> tuple[list[str], list[str]]:
    all_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and (d.startswith(prefix) if prefix else True)
    ])
    n_train = int(len(all_dirs) * 0.8)
    return all_dirs[:n_train], all_dirs[n_train:]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MRSSound+MRSSpeech+MRSDialogue → 2D az×el 균형 데이터셋',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sound-root',
                        default='./MRSAudio/MRSLife/MRSSound')
    parser.add_argument('--speech-root',
                        default='./MRSAudio/MRSSpeech/MRSSpeech')
    parser.add_argument('--dialogue-root',
                        default='./MRSAudio/MRSDialogue/MRSLife/MRSDialogue')
    parser.add_argument('--out-dir',        default='./data_mrs_max')
    parser.add_argument('--n-train',        type=int, default=8000)
    parser.add_argument('--n-val',          type=int, default=1000)
    parser.add_argument('--n-test',         type=int, default=500)
    parser.add_argument('--n-az-bins',      type=int, default=12,
                        help='Azimuth 구간 수 (12 = 30°씩)')
    parser.add_argument('--n-el-bins',      type=int, default=5,
                        help='Elevation 구간 수 (5 = 36°씩)')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--class-map-src',
                        default='./data/meta/class_map.json')
    args = parser.parse_args()

    rng      = np.random.default_rng(args.seed)
    out_dir  = os.path.abspath(args.out_dir)

    # ── 각 소스 타입별 train/test 디렉터리 분할 ───────────────────────────────
    sound_train_dirs,    sound_test_dirs    = _split_dirs(args.sound_root,    'sound')
    speech_train_dirs,   speech_test_dirs   = _split_dirs(args.speech_root,   'drama')
    dialogue_train_dirs, dialogue_test_dirs = _split_dirs(args.dialogue_root, 'dialogue')

    print('[POOL] 디렉터리 분할:')
    print(f'  MRSSound   : {len(sound_train_dirs)} train / {len(sound_test_dirs)} test')
    print(f'  MRSSpeech  : {len(speech_train_dirs)} train / {len(speech_test_dirs)} test')
    print(f'  MRSDialogue: {len(dialogue_train_dirs)} train / {len(dialogue_test_dirs)} test')

    # ── 세그먼트 로딩 ──────────────────────────────────────────────────────────
    print('\n[POOL] Train 세그먼트 로딩...')
    raw_train = (
        _load_sound_pool   (args.sound_root,    sound_train_dirs)
      + _load_speech_pool  (args.speech_root,   speech_train_dirs)
      + _load_dialogue_pool(args.dialogue_root, dialogue_train_dirs)
    )
    print(f'  로드: {len(raw_train)} segs  '
          f'(sound={sum(1 for s in raw_train if "sound/" in s["src_key"])} '
          f'speech={sum(1 for s in raw_train if "speech/" in s["src_key"])} '
          f'dialogue={sum(1 for s in raw_train if "dialogue/" in s["src_key"])})')

    print('\n[POOL] Test 세그먼트 로딩...')
    raw_test = (
        _load_sound_pool   (args.sound_root,    sound_test_dirs)
      + _load_speech_pool  (args.speech_root,   speech_test_dirs)
      + _load_dialogue_pool(args.dialogue_root, dialogue_test_dirs)
    )
    print(f'  로드: {len(raw_test)} segs')

    # ── 2D DOA 균형 가중치 부여 ───────────────────────────────────────────────
    print(f'\n[POOL] Train DOA 분포 (az×el {args.n_az_bins}×{args.n_el_bins}):')
    flat_train = build_balanced_pool(raw_train, args.n_az_bins, args.n_el_bins)

    print(f'\n[POOL] Test DOA 분포:')
    flat_test  = build_balanced_pool(raw_test,  args.n_az_bins, args.n_el_bins)

    # ── 출력 디렉터리 생성 ────────────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(out_dir, 'audio',       split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'annotations', split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'meta'), exist_ok=True)

    base_val  = args.n_train
    base_test = args.n_train + args.n_val
    split_meta = {
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
        json.dump(split_meta, f, indent=2)

    # class_map 복사
    cmap_src = os.path.abspath(args.class_map_src)
    cmap_dst = os.path.join(out_dir, 'meta', 'class_map.json')
    if os.path.exists(cmap_src):
        shutil.copy2(cmap_src, cmap_dst)
        print(f'\n[META] class_map.json 복사 완료')
    else:
        print(f'\n[META] class_map_src 없음 ({cmap_src}), 스킵')

    # ── 씬 생성 ────────────────────────────────────────────────────────────────
    print(f'\n[TRAIN] {args.n_train}개 씬 생성')
    n_train = build_split(
        flat_train, args.n_train, 0,
        os.path.join(out_dir, 'audio',       'train'),
        os.path.join(out_dir, 'annotations', 'train'),
        np.random.default_rng(args.seed),
    )

    print(f'\n[VAL]   {args.n_val}개 씬 생성')
    n_val = build_split(
        flat_train, args.n_val, base_val,
        os.path.join(out_dir, 'audio',       'val'),
        os.path.join(out_dir, 'annotations', 'val'),
        np.random.default_rng(args.seed + 1),
    )

    print(f'\n[TEST]  {args.n_test}개 씬 생성')
    n_test = build_split(
        flat_test, args.n_test, base_test,
        os.path.join(out_dir, 'audio',       'test'),
        os.path.join(out_dir, 'annotations', 'test'),
        np.random.default_rng(args.seed + 2),
        src_dist=(0.3, 0.4, 0.3),
    )

    # split.json 실제 수 반영
    split_meta['train']['n_scenes'] = n_train
    split_meta['val'  ]['n_scenes'] = n_val
    split_meta['test' ]['n_scenes'] = n_test
    with open(os.path.join(out_dir, 'meta', 'split.json'), 'w') as f:
        json.dump(split_meta, f, indent=2)

    print(f'\n[DONE]  출력: {out_dir}')
    print(f'  train: {n_train}  val: {n_val}  test: {n_test}')


if __name__ == '__main__':
    main()
