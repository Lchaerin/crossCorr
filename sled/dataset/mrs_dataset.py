#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — MRSLife/MRSSound Dataset & DataLoader
================================================

데이터 구조
-----------
MRSAudio/MRSLife/MRSSound/
  sound{NNN}/
    metadata.json          — 세그먼트 목록 (event 이름, start/stop, pos_fn 등)
    segment{MMM}.npy       — shape (T_seg, 4), dtype float64
                             col0: right_rel (m)   — 리스너 기준 오른쪽 (+)
                             col1: forward_rel (m) — 리스너 기준 앞 (+)
                             col2: up_rel (m)       — 리스너 기준 위 (+)
                             col3: time_ms          — 녹음 절대 시간 (ms)
    sound{NNN}_binaural.wav — 48 kHz 스테레오 binaural 오디오

좌표 변환 (MRS → SLED)
-----------------------
MRS 규약: 90° = 정면, 0°~180° = 앞, -180°~0° = 뒤  (리스너 기준 상대 xyz로 저장)
SLED 규약: CW, 0° = 앞, 90° = 오른쪽
  x = cos(el)·cos(az_sled)  ← 앞/뒤
  y = cos(el)·sin(az_sled)  ← 오른쪽이 양수
  z = sin(el)               ← 위가 양수

MRS (right, forward, up) → SLED unit vector:
  doa_sled = normalize( [forward, right, up] )
           = normalize( [col1, col0, col2] )

클래스 맵 (27 클래스)
  WoodenClapper / woodenclapper → 동일 클래스 24
  toy train / toy train 8       → 동일 클래스 21

분할: 208개 sound 디렉터리 중
  sound001–sound166 (80%) → train
  sound167–sound208 (20%) → val
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── 클래스 맵 ──────────────────────────────────────────────────────────────────

# 27 MRS-native 클래스 (WoodenClapper=woodenclapper, toy train 8=toy train 통합)
MRS_CLASS_MAP: dict[str, int] = {
    'bell'                    :  0,
    'birdwhistle'             :  1,
    'ClashCymbals'            :  2,
    'chinesecymbals'          :  3,
    'clap'                    :  4,
    'fanwithpaper'            :  5,
    'gong'                    :  6,
    'hairdryer'               :  7,
    'handbell'                :  8,
    'maracas'                 :  9,
    'MultiPitchPercussionTube': 10,
    'rainstick'               : 11,
    'rattle'                  : 12,
    'rotatingclapperboard'    : 13,
    'slitdrum'                : 14,
    'squeakingrubberchicken'  : 15,
    'stickbell'               : 16,
    'tambourine'              : 17,
    'Tear off tape'           : 18,
    'thundertube'             : 19,
    'toy car'                 : 20,
    'toy train'               : 21,
    'toy train 8'             : 21,  # toy train과 동일
    'triangle'                : 22,
    'whistle'                 : 23,
    'WoodenClapper'           : 24,
    'woodenclapper'           : 24,  # WoodenClapper와 동일
    'woodenfish'              : 25,
    'woodenshaker'            : 26,
}
MRS_N_CLASSES: int = 27  # 0–26

# MRS 이벤트 → FSD50K 클래스 ID (209-class 사전학습 모델 파인튜닝용)
# FSD50K에 없는 클래스는 음향적으로 가장 유사한 카테고리로 매핑
MRS_TO_FSD50K_MAP: dict[str, int] = {
    # ── 타악기 (금속) ──────────────────────────────────────────────────────────
    'bell'                    :  11,   # Bell               (직접 일치)
    'handbell'                :  11,   # Bell               (핸드벨 = 벨 악기)
    'stickbell'               :  34,   # Chime              (스틱 끝의 벨, 차임 계열 음색)
    'gong'                    :  87,   # Gong               (직접 일치)
    'ClashCymbals'            :  48,   # Crash_cymbal       (클래시 = 크래시 심벌)
    'chinesecymbals'          :  57,   # Cymbal             (심벌즈 계열)
    'triangle'                :  34,   # Chime              (삼각형 타악기 → 지속 벨 음색)

    'clap'                    :  39,   # Clapping           (직접 일치)

    # ── 타악기 (목재) ──────────────────────────────────────────────────────────
    'slitdrum'                : 111,   # Mallet_percussion  (슬릿드럼 = 말렛 타악기)
    'woodenfish'              : 111,   # Mallet_percussion  (목어 = 나무 말렛 타악기)
    'MultiPitchPercussionTube': 111,   # Mallet_percussion  (다음정 관 타악기, 붐워커 계열)
    'WoodenClapper'           :  39,   # Clapping           (나무 박자기 → 박수/타격 음색)
    'woodenclapper'           :  39,   # Clapping           (동일)

    # ── 타악기 (흔들기 / 래틀) ────────────────────────────────────────────────
    'rattle'                  : 137,   # Rattle_(instrument)(직접 일치)
    'maracas'                 : 137,   # Rattle_(instrument)(마라카스 = 흔들기 타악기)
    'woodenshaker'            : 137,   # Rattle_(instrument)(나무 셰이커)
    'tambourine'              : 165,   # Tambourine         (직접 일치)
    'rotatingclapperboard'    : 135,   # Ratchet_and_pawl   (돌림판 = 래칫 기어 음향)

    # ── 타악기 (특수 효과음) ───────────────────────────────────────────────────
    'rainstick'               : 133,   # Rain               (레인스틱 = 빗소리 재현 악기)
    'thundertube'             : 170,   # Thunder            (썬더튜브 = 천둥소리 재현)
    'birdwhistle'             :  36,   # Chirp_and_tweet    (새 울음 재현 휘슬)

    # ── 관악기 / 호흡 계열 ─────────────────────────────────────────────────────
    'whistle'                 : 194,   # Wind_instrument_and_woodwind_instrument

    # ── 마찰 / 에어 계열 ──────────────────────────────────────────────────────
    'fanwithpaper'            : 190,   # Whoosh_and_swoosh_and_swish (종이 부채 = 바람 소음)
    'hairdryer'               : 113,   # Mechanical_fan     (헤어드라이어 = 팬 모터 소음)

    # ── 고무 / 소재 계열 ──────────────────────────────────────────────────────
    'squeakingrubberchicken'  : 160,   # Squeak             (직접 일치)

    # ── 테이프 / 종이 ─────────────────────────────────────────────────────────
    'Tear off tape'           : 123,   # Packing_tape_and_duct_tape (직접 일치)

    # ── 장난감 ────────────────────────────────────────────────────────────────
    'toy car'                 :  26,   # Car                (장난감 자동차 → 자동차 음색)
    'toy train'               : 177,   # Train              (직접 일치)
    'toy train 8'             : 177,   # Train              (동일 클래스)
}


# ── 데이터셋 ───────────────────────────────────────────────────────────────────

class MRSSoundDataset(Dataset):
    """MRSLife/MRSSound binaural 오디오 데이터셋.

    Parameters
    ----------
    mrs_root      : MRSSound 디렉터리 경로
                    (…/MRSAudio/MRSLife/MRSSound)
    split         : 'train' | 'val'
    window_frames : 한 샘플의 20ms 프레임 수 (default 256 = 5.12s)
    augment_scs   : Stereo Channel Swap 증강 (train 전용)
    use_fsd50k_cls: True 이면 FSD50K 클래스 ID(0–208) 사용,
                    False(기본) 이면 MRS 클래스 ID(0–26) 사용
    min_loudness_db: 이보다 조용한 프레임은 비활성 처리

    __getitem__ returns
    -------------------
    {
      'audio'    : [2, window_frames * 960]  float32  (L, R)
      'cls'      : [window_frames, 1]        int64    (-1 = 비활성)
      'doa'      : [window_frames, 1, 3]     float32  (SLED 규약 단위벡터)
      'loud'     : [window_frames, 1]        float32  (dBFS)
      'mask'     : [window_frames, 1]        bool
      'scene_id' : str
    }
    """

    HOP_SAMPLES   = 960     # 20ms @ 48 kHz
    SAMPLE_RATE   = 48_000
    NPY_DT_MS     = 50.0    # MRS npy 어노테이션 간격 (ms)

    def __init__(
        self,
        mrs_root      : str,
        split         : str  = 'train',
        window_frames : int  = 256,
        augment_scs   : bool = True,
        use_fsd50k_cls: bool = False,
        min_loudness_db: float = -60.0,
    ):
        super().__init__()
        self.mrs_root       = mrs_root
        self.split          = split
        self.window_frames  = window_frames
        self.augment_scs    = augment_scs and (split == 'train')
        self.use_fsd50k_cls = use_fsd50k_cls
        self.min_loudness_db = min_loudness_db
        self.cls_map = MRS_TO_FSD50K_MAP if use_fsd50k_cls else MRS_CLASS_MAP

        # sound001–sound208 열거 후 train/val 분할
        all_sounds = sorted([
            d for d in os.listdir(mrs_root)
            if d.startswith('sound') and os.path.isdir(os.path.join(mrs_root, d))
        ])
        n_total = len(all_sounds)
        n_train = int(n_total * 0.8)
        if split == 'train':
            sound_list = all_sounds[:n_train]
        elif split == 'val':
            sound_list = all_sounds[n_train:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # (wav_path, npy_path, class_id, start_ms, stop_ms) 아이템 목록 구성
        self.items: list[dict] = []
        for snd in sound_list:
            snd_dir  = os.path.join(mrs_root, snd)
            meta_path = os.path.join(snd_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta_list = json.load(f)

            for seg_meta in meta_list:
                event    = seg_meta.get('event', '')
                if event not in self.cls_map:
                    continue  # 미등록 이벤트 스킵
                class_id = self.cls_map[event]

                # 절대 경로로 변환
                pos_fn_rel = seg_meta['pos_fn']            # 상대 경로 (MRSLife/…)
                # pos_fn_rel은 "MRSLife/MRSSound/soundNNN/segmentMMM.npy" 형태
                # mrs_root 의 부모 부모 (= MRSAudio) 기준으로 해석
                mrs_base = os.path.dirname(os.path.dirname(mrs_root))  # MRSAudio
                npy_path = os.path.join(mrs_base, pos_fn_rel)
                wav_fn   = seg_meta['wav_fn']              # "MRSLife/MRSSound/…/binaural.wav"
                wav_path = os.path.join(mrs_base, wav_fn)

                if not os.path.exists(npy_path) or not os.path.exists(wav_path):
                    continue

                start_ms = seg_meta['start'] * 1000.0  # seconds → ms
                stop_ms  = seg_meta['stop']  * 1000.0

                # 어노테이션 프레임 수 확인 (너무 짧으면 스킵)
                dur_frames = int((stop_ms - start_ms) / 20.0)
                if dur_frames < 4:
                    continue

                if split == 'val':
                    # val: 비중복 윈도우마다 고정 아이템 생성 (결정론적 평가)
                    stride = self.window_frames
                    t = 0
                    while t < dur_frames:
                        t0 = min(t, max(0, dur_frames - self.window_frames))
                        self.items.append({
                            'wav_path'      : wav_path,
                            'npy_path'      : npy_path,
                            'class_id'      : class_id,
                            'start_ms'      : start_ms,
                            'stop_ms'       : stop_ms,
                            'scene_id'      : f'{snd}_{seg_meta["item_name"]}_w{t0}',
                            't_start_frame' : t0,   # 고정 윈도우
                        })
                        t += stride
                        if dur_frames <= self.window_frames:
                            break  # 짧은 세그먼트는 1개만
                else:
                    # train: 랜덤 윈도우 (에포크마다 다른 구간)
                    self.items.append({
                        'wav_path'      : wav_path,
                        'npy_path'      : npy_path,
                        'class_id'      : class_id,
                        'start_ms'      : start_ms,
                        'stop_ms'       : stop_ms,
                        'scene_id'      : f'{snd}_{seg_meta["item_name"]}',
                        't_start_frame' : None,   # __getitem__에서 랜덤 선택
                    })

        if not self.items:
            raise RuntimeError(
                f"No items found for split='{split}' in {mrs_root}. "
                "Check MRSSound directory structure."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        import soundfile as sf

        item     = self.items[idx]
        cls_id   = item['class_id']
        start_ms = item['start_ms']
        stop_ms  = item['stop_ms']

        # ── npy 어노테이션 로드 ───────────────────────────────────────────────
        npy = np.load(item['npy_path'])   # (T_npy, 4): (right, fwd, up, time_ms)
        npy_times   = npy[:, 3]           # ms (절대 시간)
        npy_right   = npy[:, 0]
        npy_forward = npy[:, 1]
        npy_up      = npy[:, 2]

        # ── 세그먼트 길이 계산 & 윈도우 선택 ────────────────────────────────
        T_seg = int((stop_ms - start_ms) / 20.0)   # SLED 프레임 수
        t_start_fixed = item.get('t_start_frame')
        if t_start_fixed is not None:
            # val: 저장된 고정 윈도우 사용
            t_start_frame = t_start_fixed
        elif T_seg <= self.window_frames:
            t_start_frame = 0
        else:
            # train: 랜덤 윈도우 (augmentation 효과)
            t_start_frame = np.random.randint(0, T_seg - self.window_frames)
        t_end_frame = t_start_frame + self.window_frames

        # ── 어노테이션 → 20ms 프레임으로 선형 보간 ───────────────────────
        # SLED 프레임 중간 시간 (ms, 절대)
        frame_times_ms = start_ms + (np.arange(T_seg) + 0.5) * 20.0   # [T_seg]

        # npy 범위를 벗어나는 경우 경계값으로 클램핑
        t_min, t_max = npy_times[0], npy_times[-1]
        frame_times_clamped = np.clip(frame_times_ms, t_min, t_max)

        right_f   = np.interp(frame_times_clamped, npy_times, npy_right)    # [T_seg]
        forward_f = np.interp(frame_times_clamped, npy_times, npy_forward)
        up_f      = np.interp(frame_times_clamped, npy_times, npy_up)

        # ── 윈도우 슬라이스 ─────────────────────────────────────────────────
        def _pad_slice(arr, t0, t1, pad_val):
            sliced = arr[t0:t1]
            pad_len = self.window_frames - sliced.shape[0]
            if pad_len > 0:
                sliced = np.concatenate([sliced, np.full(pad_len, pad_val, dtype=arr.dtype)])
            return sliced

        right_w   = _pad_slice(right_f,   t_start_frame, t_end_frame, 0.0)
        forward_w = _pad_slice(forward_f, t_start_frame, t_end_frame, 0.0)
        up_w      = _pad_slice(up_f,      t_start_frame, t_end_frame, 0.0)

        # ── MRS → SLED DOA 단위벡터 변환 ────────────────────────────────────
        # MRS (right, forward, up) → SLED (x=fwd, y=right, z=up)
        doa_raw = np.stack([forward_w, right_w, up_w], axis=1)   # [T, 3]
        norm    = np.linalg.norm(doa_raw, axis=1, keepdims=True)  # [T, 1]
        norm    = np.where(norm < 1e-8, 1.0, norm)
        doa_w   = doa_raw / norm                                   # [T, 3] 정규화

        # ── 오디오 로드 (세그먼트 구간만) ────────────────────────────────────
        s_start_abs = int(start_ms  * self.SAMPLE_RATE / 1000)
        s_end_abs   = int(stop_ms   * self.SAMPLE_RATE / 1000)
        s_win_start = s_start_abs + t_start_frame * self.HOP_SAMPLES
        s_win_end   = s_win_start  + self.window_frames * self.HOP_SAMPLES

        try:
            audio_np, _ = sf.read(
                item['wav_path'],
                start  = s_win_start,
                stop   = s_win_end,
                dtype  = 'float32',
                always_2d = True,
            )
        except Exception:
            # 범위 초과 등 예외 → 무음으로 패딩
            audio_np = np.zeros((self.window_frames * self.HOP_SAMPLES, 2), dtype=np.float32)

        # 오디오 패딩 (짧은 세그먼트)
        n_expected = self.window_frames * self.HOP_SAMPLES
        if audio_np.shape[0] < n_expected:
            pad = np.zeros((n_expected - audio_np.shape[0], 2), dtype=np.float32)
            audio_np = np.concatenate([audio_np, pad], axis=0)
        audio_np = audio_np[:n_expected]   # [N, 2]
        audio_np = audio_np.T              # [2, N]

        # ── 오디오 피크 정규화 (MRS 녹음은 절대 음압이 낮음) ────────────────
        peak = np.abs(audio_np).max()
        if peak > 1e-6:
            audio_np = audio_np / peak * 0.5   # 피크를 -6 dBFS 로 정규화

        # ── Random gain 증강 (train 전용, ±6 dB) ─────────────────────────
        if self.split == 'train':
            gain_db = np.random.uniform(-6.0, 6.0)
            audio_np = audio_np * (10.0 ** (gain_db / 20.0))

        # ── 음량(loudness) 계산 (dBFS, per 20ms frame) ──────────────────────
        mono     = audio_np.mean(0)                          # [N]
        frames_a = mono.reshape(self.window_frames, self.HOP_SAMPLES)
        rms      = np.sqrt(np.mean(frames_a ** 2, axis=1))  # [T]
        loud_w   = 20.0 * np.log10(rms + 1e-8)              # [T] dBFS

        # ── 마스크: 소리가 존재하고 충분히 큰 프레임 ──────────────────────
        mask_w = loud_w > self.min_loudness_db    # [T]
        cls_w  = np.full(self.window_frames, cls_id, dtype=np.int64)
        cls_w[~mask_w] = -1  # 비활성 프레임 센티널

        # shape: [T] → [T, 1] (슬롯 차원 추가)
        cls_w  = cls_w[:, None]            # [T, 1]
        doa_w  = doa_w[:, None, :]        # [T, 1, 3]
        loud_w = loud_w[:, None]           # [T, 1]
        mask_w = mask_w[:, None]           # [T, 1]

        # ── torch 변환 ────────────────────────────────────────────────────────
        audio_t = torch.from_numpy(audio_np)                        # [2, N]
        cls_t   = torch.from_numpy(cls_w.astype(np.int64))          # [T, 1]
        doa_t   = torch.from_numpy(doa_w.astype(np.float32))        # [T, 1, 3]
        loud_t  = torch.from_numpy(loud_w.astype(np.float32))       # [T, 1]
        mask_t  = torch.from_numpy(mask_w.astype(bool))             # [T, 1]

        # ── Stereo Channel Swap 증강 ─────────────────────────────────────────
        if self.augment_scs and torch.rand(1).item() < 0.5:
            audio_t = audio_t.flip(0)
            # SLED 규약: y 성분 부호 반전 = 방위각 반전
            doa_t = doa_t.clone()
            doa_t[..., 1] = -doa_t[..., 1]

        return {
            'audio'   : audio_t,
            'cls'     : cls_t,
            'doa'     : doa_t,
            'loud'    : loud_t,
            'mask'    : mask_t,
            'scene_id': item['scene_id'],
        }


# ── DataLoader 팩토리 ──────────────────────────────────────────────────────────

def build_mrs_dataloader(
    mrs_root      : str,
    split         : str,
    batch_size    : int   = 8,
    window_frames : int   = 256,
    augment_scs   : bool  = True,
    num_workers   : int   = 4,
    use_fsd50k_cls: bool  = False,
    min_loudness_db: float = -60.0,
    **kwargs,
) -> DataLoader:
    """MRSSoundDataset용 DataLoader 생성."""
    shuffle = kwargs.pop('shuffle', split == 'train')
    pin_memory = kwargs.pop('pin_memory', True)
    dataset = MRSSoundDataset(
        mrs_root       = mrs_root,
        split          = split,
        window_frames  = window_frames,
        augment_scs    = augment_scs,
        use_fsd50k_cls = use_fsd50k_cls,
        min_loudness_db = min_loudness_db,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = (split == 'train'),
        **kwargs,
    )
