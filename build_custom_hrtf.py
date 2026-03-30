#!/usr/bin/env python3
"""
build_custom_hrtf.py
====================
MRS binaural 녹음의 실제 공간 단서와 가장 잘 맞는 HRTF를 선별하거나,
방향(direction)별로 가장 적합한 HRTF 값을 골라 합성 HRTF를 만든다.

원리
----
MRS 데이터셋은 binaural 오디오와 정확한 DOA(npy)를 함께 제공한다.
각 방향에서 측정된 실제 Cross-Spectral Density(CSD = X_R × X_L*)와
각 피험자 HRTF의 cross-spectrum(W = H_R × H_L*)을 비교하여
어떤 HRTF가 그 방향에서 실제 녹음과 가장 유사한지 스코어링한다.

출력
----
1. 피험자별 전체 평균 스코어 (→ 단일 최적 HRTF 순위)
2. 방향 bin별 최적 피험자 지도 (coverage 시각화)
3. 방향별 winner를 합친 custom SOFA 파일 (→ hrtf/custom_mrs.sofa)

Usage
-----
    python build_custom_hrtf.py \\
        --hrtf-dir  ./hrtf \\
        --mrs-root  ./MRSAudio/MRSLife/MRSSound \\
        --out-sofa  ./hrtf/custom_mrs.sofa \\
        --n-segs    300        # 평가에 사용할 MRS 세그먼트 수
        --min-dur   2.0        # 최소 세그먼트 길이(초)
"""

import argparse
import json
import os
import glob
import random

import numpy as np
import soundfile as sf
import h5py
from scipy.signal import stft as scipy_stft
from tqdm import tqdm

# ── 상수 ──────────────────────────────────────────────────────────────────────
SR        = 48_000
N_FFT     = 2048
HOP       = 960
N_MELS    = 64
FMIN      = 20.0
FMAX      = 16_000.0
N_FRAMES  = 8      # CSD 평균에 사용할 STFT 프레임 수
MIN_RMS   = 1e-4   # 무음 구간 스킵 임계값


# =============================================================================
# SOFA 로드 유틸
# =============================================================================

def load_sofa_crossspec(sofa_path: str):
    """SOFA → W[M, F] = H_R[M,F] * conj(H_L[M,F]) (복소수)

    Returns
    -------
    W       : [M, F_used] complex64  (F_used = FFT bins that cover fmin~fmax)
    pos_az  : [M] float32  방위각 (도, SOFA CCW: 0=앞, 90=왼쪽)
    pos_el  : [M] float32  고도각 (도)
    freqs   : [F_used] float32
    """
    with h5py.File(sofa_path, 'r') as f:
        ir  = f['Data.IR'][:]          # [M, 2, N]
        pos = f['SourcePosition'][:]   # [M, 3] (az, el, dist)

    M, _, N = ir.shape
    # FFT
    H  = np.fft.rfft(ir, n=N_FFT, axis=-1)  # [M, 2, F]
    H_L, H_R = H[:, 0, :], H[:, 1, :]       # [M, F]

    freqs_all = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    freq_mask = (freqs_all >= FMIN) & (freqs_all <= FMAX)

    W      = (H_R[:, freq_mask] * np.conj(H_L[:, freq_mask])).astype(np.complex64)
    pos_az = pos[:, 0].astype(np.float32)
    pos_el = pos[:, 1].astype(np.float32)
    freqs  = freqs_all[freq_mask].astype(np.float32)

    return W, pos_az, pos_el, freqs


def load_all_hrtfs(hrtf_dir: str):
    """모든 SOFA 파일 로드 → W_all[S, M, F]"""
    sofa_paths = sorted(glob.glob(os.path.join(hrtf_dir, 'p*.sofa')))
    if not sofa_paths:
        raise FileNotFoundError(f'No p*.sofa files in {hrtf_dir}')

    print(f'[HRTF] {len(sofa_paths)}개 SOFA 로딩...')
    W_list = []
    for path in tqdm(sofa_paths, desc='load SOFA', ncols=70):
        W, az, el, freqs = load_sofa_crossspec(path)
        W_list.append(W)

    W_all = np.stack(W_list, axis=0)  # [S, M, F]
    print(f'       W_all: {W_all.shape}  (subjects × directions × freqs)')
    return W_all, az, el, freqs, sofa_paths


# =============================================================================
# MRS 세그먼트 수집
# =============================================================================

def collect_mrs_segments(mrs_root: str, min_dur: float = 2.0):
    """MRS 세그먼트 목록 반환. [{wav_path, npy_path, start_ms, stop_ms}]"""
    segs = []
    for meta_path in glob.glob(os.path.join(mrs_root, '*/metadata.json')):
        with open(meta_path) as f:
            items = json.load(f)
        wav_base = os.path.dirname(meta_path)
        for item in items:
            wav_fn = item.get('wav_fn', '')
            pos_fn = item.get('pos_fn', '')
            if not wav_fn or not pos_fn:
                continue
            mrs_base = os.path.dirname(os.path.dirname(mrs_root))
            wav_path = os.path.join(mrs_base, wav_fn)
            npy_path = os.path.join(mrs_base, pos_fn)
            if not os.path.exists(wav_path) or not os.path.exists(npy_path):
                continue
            start_ms = item['start'] * 1000.0
            stop_ms  = item['stop']  * 1000.0
            if (stop_ms - start_ms) < min_dur * 1000.0:
                continue
            segs.append({'wav_path': wav_path,
                         'npy_path': npy_path,
                         'start_ms': start_ms,
                         'stop_ms':  stop_ms})
    return segs


# =============================================================================
# 실제 CSD 계산
# =============================================================================

def compute_measured_csd(wav_path: str, start_ms: float, stop_ms: float,
                          freqs: np.ndarray):
    """MRS binaural 오디오 → 측정 CSD [F_used] complex64

    N_FRAMES개 STFT 프레임을 균등 샘플링해서 평균
    """
    s_start = int(start_ms * SR / 1000)
    s_stop  = int(stop_ms  * SR / 1000)
    try:
        audio, file_sr = sf.read(wav_path, start=s_start, stop=s_stop,
                                  dtype='float32', always_2d=True)
    except Exception:
        return None

    if audio.shape[0] < N_FFT or audio.shape[1] < 2:
        return None

    audio = audio.T  # [2, N]
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < MIN_RMS:
        return None

    # STFT
    _, _, ZL = scipy_stft(audio[0], fs=SR, nperseg=N_FFT, noverlap=N_FFT-HOP,
                           window='hann', padded=False)
    _, _, ZR = scipy_stft(audio[1], fs=SR, nperseg=N_FFT, noverlap=N_FFT-HOP,
                           window='hann', padded=False)
    # ZL, ZR: [F, T_stft]

    T_stft = ZL.shape[1]
    if T_stft < N_FRAMES:
        frame_idxs = np.arange(T_stft)
    else:
        frame_idxs = np.linspace(0, T_stft - 1, N_FRAMES, dtype=int)

    ZL_sel = ZL[:, frame_idxs]   # [F, N_FRAMES]
    ZR_sel = ZR[:, frame_idxs]

    # Cross-spectrum per frame, then average
    csd_all = ZR_sel * np.conj(ZL_sel)          # [F, N_FRAMES]
    csd_avg = csd_all.mean(axis=1)               # [F]

    # rfft frequency mask
    freqs_stft = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    freq_mask  = (freqs_stft >= FMIN) & (freqs_stft <= FMAX)
    csd_used   = csd_avg[freq_mask].astype(np.complex64)  # [F_used]

    return csd_used


# =============================================================================
# 방향 매칭
# =============================================================================

def find_closest_direction(az_query: float, el_query: float,
                            az_all: np.ndarray, el_all: np.ndarray) -> int:
    """SOFA 방향 목록에서 가장 가까운 인덱스 반환 (각도 거리)"""
    # 구면 거리 (내적 근사)
    az_q_r = np.radians(az_query)
    el_q_r = np.radians(el_query)
    az_r   = np.radians(az_all)
    el_r   = np.radians(el_all)

    dot = (np.cos(el_q_r) * np.cos(el_r) * np.cos(az_q_r - az_r)
           + np.sin(el_q_r) * np.sin(el_r))
    dot = np.clip(dot, -1.0, 1.0)
    return int(np.argmax(dot))   # 가장 가까운 = 내적 최대


def doa_sled_to_sofa(doa_vec: np.ndarray):
    """SLED 단위벡터 [fwd, right, up] → SOFA 방위각/고도각 (도)
    SOFA: az CCW (0=front, 90=left), el
    SLED: x=fwd, y=right, z=up  →  az_cw = atan2(y,x)  → az_sofa = -az_cw
    """
    x, y, z = doa_vec
    el = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
    az_cw  = float(np.degrees(np.arctan2(y, x)) % 360.0)
    az_sofa = (-az_cw) % 360.0
    return az_sofa, el


# =============================================================================
# 스코어링
# =============================================================================

def cosine_similarity_complex(a: np.ndarray, b: np.ndarray) -> float:
    """복소수 벡터 코사인 유사도 (실수 반환)"""
    dot  = np.real(np.dot(a, np.conj(b)))
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(dot / norm)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MRS 데이터 기반 최적 HRTF 선별 및 방향별 합성',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hrtf-dir',  default='./hrtf')
    parser.add_argument('--mrs-root',  default='./MRSAudio/MRSLife/MRSSound')
    parser.add_argument('--out-sofa',  default='./hrtf/custom_mrs.sofa')
    parser.add_argument('--n-segs',    type=int, default=300,
                        help='평가에 사용할 MRS 세그먼트 수 (랜덤 샘플링)')
    parser.add_argument('--min-dur',   type=float, default=2.0,
                        help='최소 세그먼트 길이 (초)')
    parser.add_argument('--top-k',     type=int, default=5,
                        help='coverage 미달 bin에 채울 전체 순위 상위 K개 사용')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # ── 1. HRTF 로드 ──────────────────────────────────────────────────────────
    W_all, sofa_az, sofa_el, freqs, sofa_paths = load_all_hrtfs(args.hrtf_dir)
    S, M, F = W_all.shape
    print(f'       {S} subjects, {M} directions, {F} freq bins\n')

    # W 정규화 (방향별) - 코사인 유사도 계산 속도 향상
    W_norm = W_all / (np.linalg.norm(W_all, axis=-1, keepdims=True) + 1e-12)

    # ── 2. MRS 세그먼트 수집 ──────────────────────────────────────────────────
    print('[MRS]  세그먼트 수집...')
    all_segs = collect_mrs_segments(args.mrs_root, min_dur=args.min_dur)
    print(f'       전체 {len(all_segs)}개 세그먼트')

    if len(all_segs) > args.n_segs:
        idxs = rng.choice(len(all_segs), args.n_segs, replace=False)
        segs = [all_segs[i] for i in idxs]
    else:
        segs = all_segs
    print(f'       평가 사용: {len(segs)}개\n')

    # ── 3. 방향별 스코어 누적 ─────────────────────────────────────────────────
    # dir_scores[m, s] = 방향 m에서 피험자 s의 누적 유사도
    # dir_count[m]     = 방향 m에서 누적된 샘플 수
    dir_scores = np.zeros((M, S), dtype=np.float64)
    dir_count  = np.zeros(M,      dtype=np.int32)

    print('[EVAL] 방향별 스코어 계산...')
    skipped = 0
    for seg in tqdm(segs, desc='scoring', ncols=70):
        # DOA 로드 (npy 중간 프레임)
        try:
            npy = np.load(seg['npy_path'])  # [T_npy, 4]: (right, fwd, up, time_ms)
        except Exception:
            skipped += 1
            continue

        # 세그먼트 중간 시각의 DOA 사용
        mid_ms = (seg['start_ms'] + seg['stop_ms']) / 2.0
        times  = npy[:, 3]
        idx    = int(np.argmin(np.abs(times - mid_ms)))
        doa_right, doa_fwd, doa_up = npy[idx, 0], npy[idx, 1], npy[idx, 2]
        doa_vec = np.array([doa_fwd, doa_right, doa_up], dtype=np.float32)
        norm    = np.linalg.norm(doa_vec)
        if norm < 1e-6:
            skipped += 1
            continue
        doa_vec /= norm

        az_sofa, el_sofa = doa_sled_to_sofa(doa_vec)

        # 가장 가까운 SOFA 방향 인덱스
        m_idx = find_closest_direction(az_sofa, el_sofa, sofa_az, sofa_el)

        # 실제 CSD 계산
        csd = compute_measured_csd(seg['wav_path'],
                                    seg['start_ms'], seg['stop_ms'], freqs)
        if csd is None:
            skipped += 1
            continue

        csd_n = csd / (np.linalg.norm(csd) + 1e-12)  # 정규화

        # S개 피험자 모두와 코사인 유사도 (vectorized)
        sims = np.real(W_norm[:, m_idx, :] @ np.conj(csd_n))  # [S]

        dir_scores[m_idx] += sims
        dir_count[m_idx]  += 1

    print(f'       스킵: {skipped}개 / {len(segs)}개\n')

    # ── 4. 피험자별 전체 순위 ──────────────────────────────────────────────────
    covered_dirs = (dir_count > 0)
    n_covered    = int(covered_dirs.sum())
    print(f'[RESULT] 커버된 방향 bin: {n_covered} / {M}')

    # 전체 평균 스코어 (커버된 방향만)
    global_scores = np.zeros(S)
    for s in range(S):
        if n_covered > 0:
            global_scores[s] = dir_scores[covered_dirs, s].mean()

    rank_order = np.argsort(-global_scores)  # 내림차순

    print('\n  순위  파일명                    평균 유사도')
    print('  ' + '-' * 50)
    for i, s_idx in enumerate(rank_order[:20]):
        name = os.path.basename(sofa_paths[s_idx])
        print(f'  {i+1:>4}  {name:<28}  {global_scores[s_idx]:.6f}')

    best_global_idx = rank_order[0]
    print(f'\n  ★ 단일 최적 HRTF: {os.path.basename(sofa_paths[best_global_idx])}')

    # ── 5. 방향별 winner 결정 ─────────────────────────────────────────────────
    top_k_subjects = rank_order[:args.top_k]  # 전체 상위 K개 (fallback용)

    # winner_map[m]: 방향 m의 최적 피험자 인덱스
    winner_map = np.full(M, best_global_idx, dtype=np.int32)  # 기본값: 전체 1위

    for m in range(M):
        if dir_count[m] > 0:
            winner_map[m] = int(np.argmax(dir_scores[m]))
        # coverage 0인 bin은 전체 1위로 유지

    # winner 분포 통계
    unique, counts = np.unique(winner_map, return_counts=True)
    print(f'\n  방향별 winner 분포 (상위 10개 피험자):')
    top10 = sorted(zip(counts, unique), reverse=True)[:10]
    for cnt, s_idx in top10:
        print(f'    {os.path.basename(sofa_paths[s_idx]):<28} {cnt:>4}개 방향')

    # ── 6. Custom SOFA 생성 ────────────────────────────────────────────────────
    print(f'\n[SOFA]  custom HRTF 생성: {args.out_sofa}')

    # 템플릿으로 전체 1위 SOFA 사용 (방향 그리드·메타데이터 복사)
    template_path = sofa_paths[best_global_idx]

    # 각 방향별로 IR 선택
    # ir_all[s, m, ear, n] 로드는 메모리가 크므로 방향별로 순차 처리
    print('       각 방향의 IR 로딩...')

    # 먼저 필요한 피험자별 IR을 모두 메모리에 올림
    needed_subjects = set(winner_map.tolist())
    ir_by_subject   = {}
    for s_idx in tqdm(sorted(needed_subjects), desc='load IR', ncols=70):
        with h5py.File(sofa_paths[s_idx], 'r') as f:
            ir_by_subject[s_idx] = f['Data.IR'][:]  # [M, 2, N]

    # custom IR 조립
    with h5py.File(sofa_paths[0], 'r') as f_ref:
        N_ir = f_ref['Data.IR'].shape[2]
    custom_ir = np.zeros((M, 2, N_ir), dtype=np.float32)
    for m in range(M):
        s_idx = winner_map[m]
        custom_ir[m] = ir_by_subject[s_idx][m]

    # SOFA 저장 (템플릿 복사 후 IR만 교체)
    import shutil
    shutil.copy2(template_path, args.out_sofa)
    with h5py.File(args.out_sofa, 'r+') as f:
        del f['Data.IR']
        f.create_dataset('Data.IR', data=custom_ir.astype(np.float64),
                         compression='gzip')
        # Comment 업데이트
        if 'Comment' in f.attrs:
            del f.attrs['Comment']
        f.attrs['Comment'] = (
            f'Custom MRS-adapted HRTF: per-direction best-match '
            f'selected from {S} SONICOM subjects. '
            f'Template: {os.path.basename(template_path)}'
        )

    print(f'       저장 완료: {args.out_sofa}')

    # ── 7. 결과 저장 ──────────────────────────────────────────────────────────
    result = {
        'global_ranking': [
            {'rank': i+1,
             'file': os.path.basename(sofa_paths[s_idx]),
             'score': float(global_scores[s_idx])}
            for i, s_idx in enumerate(rank_order)
        ],
        'n_covered_dirs': n_covered,
        'n_total_dirs':   M,
        'n_segments_used': len(segs) - skipped,
        'best_single': os.path.basename(sofa_paths[best_global_idx]),
        'custom_sofa': args.out_sofa,
    }
    result_path = os.path.splitext(args.out_sofa)[0] + '_ranking.json'
    import json as _json
    with open(result_path, 'w') as f:
        _json.dump(result, f, indent=2)
    print(f'       순위 저장: {result_path}')

    print('\n[DONE]')
    print(f'  단일 최적 HRTF  : hrtf/{os.path.basename(sofa_paths[best_global_idx])}')
    print(f'  방향별 합성 HRTF: {args.out_sofa}')


if __name__ == '__main__':
    main()
