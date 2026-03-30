#!/home/rllab/anaconda3/bin/python
"""
Binaural Audio Scene Generator
================================
HRTF(p0001.sofa)와 효과음(.mp3)을 이용해 30초짜리 바이노럴 오디오를 합성합니다.
- 한 씬에 최대 4개의 음원이 동시에 재생될 수 있습니다.
- Ground truth(언제, 어느 위치에서 어떤 소리가 재생되는지)를 JSON으로 저장합니다.

사용법:
    python generate_audio.py                      # 기본 씬 1개 생성
    python generate_audio.py --num-scenes 5       # 씬 5개 생성
    python generate_audio.py --output-dir ./data  # 저장 폴더 지정
"""

import numpy as np
import json
import os
import argparse

from scipy.signal import fftconvolve
import soundfile as sf
import librosa

# ── 기본 경로 ─────────────────────────────────────────────────
SOFA_PATH   = os.path.join(os.path.dirname(__file__), 'hrtf', 'custom_mrs.sofa')
SFX_DIR     = os.path.join(os.path.dirname(__file__), 'soud_effects')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'output')

# ── 합성 파라미터 ─────────────────────────────────────────────
DURATION          = 30.0   # 씬 길이 (초)
FS                = 44100  # 샘플링 레이트 (HRTF 기준)
MAX_SIMULTANEOUS  = 1      # 동시 최대 음원 수
NUM_EVENTS_RANGE  = (8, 14)  # 씬당 이벤트 수
MIN_EVENT_DUR     = 1.5    # 이벤트 최소 길이 (초)
MAX_EVENT_DUR     = 8.0    # 이벤트 최대 길이 (초)
FADE_DURATION     = 0.05   # fade in/out 길이 (초)
NOISE_SNR_DB_RANGE = (20, 40)  # 배경 white noise SNR 범위 (dB), 씬마다 랜덤 선택


# ============================================================
# 1. SOFA 읽기
# ============================================================

def load_sofa(filepath):
    """SOFA 파일에서 HRIR을 읽어 반환합니다."""
    try:
        import netCDF4
        ds = netCDF4.Dataset(filepath, 'r')
        ir_data    = np.array(ds.variables['Data.IR'][:])         # (M, 2, N)
        source_pos = np.array(ds.variables['SourcePosition'][:])  # (M, 3)
        fs_hrtf    = float(ds.variables['Data.SamplingRate'][:].flat[0])
        ds.close()
    except Exception:
        import h5py
        with h5py.File(filepath, 'r') as f:
            ir_data    = np.array(f['Data.IR'])
            source_pos = np.array(f['SourcePosition'])
            fs_hrtf    = float(np.array(f['Data.SamplingRate']).flat[0])

    hrir_l    = ir_data[:, 0, :]   # (M, N)
    hrir_r    = ir_data[:, 1, :]
    azimuths  = source_pos[:, 0]   # (M,)
    elevations = source_pos[:, 1]
    print(f"[SOFA] {len(azimuths)}방향, IR길이={hrir_l.shape[1]}, fs={fs_hrtf}Hz")
    return hrir_l, hrir_r, azimuths, elevations, fs_hrtf


# ============================================================
# 2. 효과음 로드
# ============================================================

def load_sfx(sfx_dir, target_fs):
    """폴더 내 모든 mp3를 target_fs로 리샘플링해 dict로 반환합니다."""
    sfx = {}
    files = sorted([f for f in os.listdir(sfx_dir) if f.lower().endswith('.mp3')])
    print(f"[SFX] {len(files)}개 파일 로딩 중...")
    for fname in files:
        path = os.path.join(sfx_dir, fname)
        audio, _ = librosa.load(path, sr=target_fs, mono=True)
        sfx[fname] = audio
        print(f"  {fname}: {len(audio)/target_fs:.1f}초")
    return sfx


# ============================================================
# 3. 이벤트 스케줄링 (동시 제약 포함)
# ============================================================

def schedule_events(sfx, azimuths, elevations, n_samples, rng, fs):
    """
    동시 최대 MAX_SIMULTANEOUS 제약 하에 이벤트를 무작위 배치합니다.

    Returns
    -------
    list of dict : 각 이벤트 {'file', 'start_sample', 'end_sample',
                              'start_time', 'end_time', 'azimuth', 'elevation',
                              'az_idx', 'gain', 'audio_segment'}
    """
    sfx_names = list(sfx.keys())
    activity  = np.zeros(n_samples, dtype=np.int8)  # 동시 재생 카운트
    events    = []
    n_events  = rng.randint(*NUM_EVENTS_RANGE)

    for _attempt in range(n_events * 10):
        if len(events) >= n_events:
            break

        # 이벤트 길이 및 시작 시간
        dur  = rng.uniform(MIN_EVENT_DUR, MAX_EVENT_DUR)
        t_max = DURATION - dur
        if t_max <= 0:
            continue
        t_start = rng.uniform(0, t_max)
        t_end   = t_start + dur

        s_start = int(t_start * fs)
        s_end   = int(t_end   * fs)

        # 동시 재생 제약 확인
        if activity[s_start:s_end].max() >= MAX_SIMULTANEOUS:
            continue

        # 음원 파일 선택 및 구간 추출
        fname = rng.choice(sfx_names)
        src   = sfx[fname]
        n_seg = s_end - s_start

        if len(src) < n_seg:
            reps = int(np.ceil(n_seg / len(src)))
            seg  = np.tile(src, reps)[:n_seg].copy()
        else:
            offset = rng.randint(0, len(src) - n_seg)
            seg    = src[offset : offset + n_seg].copy()

        # Fade in/out
        fade = int(FADE_DURATION * fs)
        fade = min(fade, n_seg // 4)
        seg[:fade]  *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)

        # 정규화
        peak = np.max(np.abs(seg))
        if peak > 1e-8:
            seg /= peak

        # 방위각 선택
        az_idx = rng.randint(0, len(azimuths))
        gain   = rng.uniform(0.3, 0.8)

        events.append({
            'file'         : fname,
            'start_sample' : s_start,
            'end_sample'   : s_end,
            'start_time'   : round(float(t_start), 4),
            'end_time'     : round(float(t_end),   4),
            'azimuth'      : float(azimuths[az_idx]),
            'elevation'    : float(elevations[az_idx]),
            'az_idx'       : int(az_idx),
            'gain'         : round(float(gain), 4),
            'audio_segment': seg
        })
        activity[s_start:s_end] += 1

    return events


# ============================================================
# 4. 바이노럴 믹싱
# ============================================================

def mix_binaural(events, hrir_l, hrir_r, n_samples, rng):
    """이벤트들을 HRTF로 컨볼루션해 바이노럴 스테레오 믹스를 반환합니다.
    씬마다 랜덤한 SNR의 배경 white noise를 추가합니다."""
    mix_L = np.zeros(n_samples, dtype=np.float64)
    mix_R = np.zeros(n_samples, dtype=np.float64)

    for ev in events:
        seg     = ev['audio_segment']
        idx     = ev['az_idx']
        gain    = ev['gain']
        s_start = ev['start_sample']
        s_end   = ev['end_sample']
        n_seg   = s_end - s_start

        hl = hrir_l[idx, :]
        hr = hrir_r[idx, :]

        sig_l = fftconvolve(seg, hl, mode='full')[:n_seg]
        sig_r = fftconvolve(seg, hr, mode='full')[:n_seg]

        mix_L[s_start:s_end] += gain * sig_l
        mix_R[s_start:s_end] += gain * sig_r

    # 클리핑 방지 정규화
    peak = max(np.max(np.abs(mix_L)), np.max(np.abs(mix_R)))
    if peak > 1e-8:
        scale = 0.9 / peak
        mix_L *= scale
        mix_R *= scale

    # 배경 white noise 추가 (SNR은 씬마다 랜덤)
    snr_db    = rng.uniform(*NOISE_SNR_DB_RANGE)
    sig_power = np.mean(mix_L ** 2 + mix_R ** 2) / 2
    if sig_power > 1e-12:
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise_std   = np.sqrt(noise_power)
        mix_L += rng.randn(n_samples) * noise_std
        mix_R += rng.randn(n_samples) * noise_std
        print(f"  [노이즈] 배경 white noise SNR={snr_db:.1f}dB 추가")

    return mix_L, mix_R


# ============================================================
# 5. 저장
# ============================================================

def save_scene(name, mix_L, mix_R, events, output_dir, fs):
    """WAV와 Ground Truth JSON을 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)

    wav_path  = os.path.join(output_dir, f'{name}.wav')
    json_path = os.path.join(output_dir, f'{name}_gt.json')

    # WAV 저장 (float32 stereo)
    stereo = np.stack([mix_L, mix_R], axis=1).astype(np.float32)
    sf.write(wav_path, stereo, int(fs))
    print(f"[저장] 오디오: {wav_path}")

    # Ground truth JSON (audio_segment 제외하고 직렬화)
    gt_events = [
        {k: v for k, v in ev.items() if k != 'audio_segment'}
        for ev in events
    ]
    gt = {
        'scene_name'  : name,
        'duration_sec': DURATION,
        'sample_rate' : int(fs),
        'audio_file'  : f'{name}.wav',
        'num_events'  : len(events),
        'max_simultaneous': MAX_SIMULTANEOUS,
        'events'      : sorted(gt_events, key=lambda e: e['start_time'])
    }
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(gt, fp, indent=2, ensure_ascii=False)
    print(f"[저장] Ground Truth: {json_path}")
    print(f"  총 이벤트 수: {len(events)}")
    for ev in sorted(gt_events, key=lambda e: e['start_time']):
        print(f"  [{ev['start_time']:5.1f}s ~ {ev['end_time']:5.1f}s] "
              f"az={ev['azimuth']:6.1f}° el={ev['elevation']:5.1f}°  "
              f"gain={ev['gain']:.2f}  {ev['file']}")

    return wav_path, json_path


# ============================================================
# 6. 메인
# ============================================================

def synthesize_scene(name, hrir_l, hrir_r, azimuths, elevations, sfx,
                     output_dir, fs, seed=None):
    """씬 하나를 합성해 저장합니다."""
    rng = np.random.RandomState(seed)

    n_samples = int(DURATION * fs)
    events    = schedule_events(sfx, azimuths, elevations, n_samples, rng, fs)
    mix_L, mix_R = mix_binaural(events, hrir_l, hrir_r, n_samples, rng)
    return save_scene(name, mix_L, mix_R, events, output_dir, fs)


def main():
    parser = argparse.ArgumentParser(description='HRTF 바이노럴 오디오 씬 생성기')
    parser.add_argument('--num-scenes',  type=int, default=1,
                        help='생성할 씬 수 (기본: 1)')
    parser.add_argument('--output-dir',  type=str, default=OUTPUT_DIR,
                        help=f'저장 폴더 (기본: {OUTPUT_DIR})')
    parser.add_argument('--seed',        type=int, default=42,
                        help='랜덤 시드 (기본: 42)')
    args = parser.parse_args()

    print("=" * 60)
    print("  HRTF 바이노럴 오디오 씬 생성기")
    print("=" * 60)

    # 공통 리소스 로드
    hrir_l, hrir_r, azimuths, elevations, fs_hrtf = load_sofa(SOFA_PATH)
    fs = int(fs_hrtf)  # HRTF의 샘플링 레이트를 기준으로 사용
    if abs(fs_hrtf - FS) > 1:
        print(f"[리샘플] HRTF fs={fs}Hz ≠ 기본 FS={FS}Hz → 효과음을 {fs}Hz로 리샘플링합니다.")
    sfx = load_sfx(SFX_DIR, fs)

    for i in range(args.num_scenes):
        name = f'scene_{i+1:02d}'
        seed = args.seed + i
        print(f"\n{'─'*40}")
        print(f"  씬 합성 중: {name} (seed={seed})")
        print(f"{'─'*40}")
        synthesize_scene(name, hrir_l, hrir_r, azimuths, elevations,
                         sfx, args.output_dir, fs=fs, seed=seed)

    print("\n완료!")


if __name__ == '__main__':
    main()
