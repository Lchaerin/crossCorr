#!/home/rllab/anaconda3/bin/python
"""
Sound Source Localization (SSL) using Cross-Channel HRTF Method
===============================================================

Cross-channel 기법을 이용한 음원 방향 추정 알고리즘.
컨볼루션의 교환법칙을 이용하여 미지의 원본 신호 S를 소거하고,
HRTF 데이터베이스와의 교차 상관을 통해 방위각(azimuth)과 
고도각(elevation)을 추정합니다.

원리:
    실제 방향 k에서 온 소리의 바이노럴 신호:
        L(f) = S(f) * HRTF_L(k, f)
        R(f) = S(f) * HRTF_R(k, f)
    
    후보 방향 j에 대해 교차 곱셈:
        sig1 = L(f) * HRTF_R(j, f)  = S(f) * HRTF_L(k, f) * HRTF_R(j, f)
        sig2 = R(f) * HRTF_L(j, f)  = S(f) * HRTF_R(k, f) * HRTF_L(j, f)
    
    j == k 이면 sig1 == sig2 → 상관 최대

사용법:
    python ssl_cross_channel.py --sofa <HRTF.sofa> --audio <binaural.wav>

필요 라이브러리:
    - numpy, scipy, matplotlib (기본)
    - netCDF4 또는 h5py (SOFA 파일 읽기, 없으면 scipy.io.netcdf로 시도)
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')  # 헤드리스 렌더링 (MP4 저장용)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import argparse
import sys
import os
import subprocess
import threading


# ============================================================
# 1. SOFA 파일 읽기
# ============================================================

def read_sofa(filepath):
    """
    SOFA 파일에서 HRIR 데이터와 좌표를 읽어옵니다.
    
    Parameters
    ----------
    filepath : str
        .sofa 파일 경로
    
    Returns
    -------
    hrir_l : ndarray, shape (N_directions, N_samples)
        좌측 HRIR
    hrir_r : ndarray, shape (N_directions, N_samples)
        우측 HRIR
    azimuths : ndarray, shape (N_directions,)
        각 방향의 방위각 (도)
    elevations : ndarray, shape (N_directions,)
        각 방향의 고도각 (도)
    fs_hrtf : float
        HRIR 샘플링 레이트
    """
    
    # 방법 1: netCDF4 (SOFA는 보통 netCDF4/HDF5 형식)
    try:
        import netCDF4
        ds = netCDF4.Dataset(filepath, 'r')
        
        # Data.IR: (M, R, N) = (방향 수, 수신기 수(2), 샘플 수)
        ir_data = np.array(ds.variables['Data.IR'][:])
        source_pos = np.array(ds.variables['SourcePosition'][:])  # (M, 3): az, el, dist
        fs_hrtf = float(ds.variables['Data.SamplingRate'][:].flat[0])
        
        hrir_l = ir_data[:, 0, :]  # 좌측 (receiver 0)
        hrir_r = ir_data[:, 1, :]  # 우측 (receiver 1)
        azimuths = source_pos[:, 0]
        elevations = source_pos[:, 1]
        
        ds.close()
        print(f"[SOFA] netCDF4로 읽기 성공")
        print(f"  방향 수: {len(azimuths)}, HRIR 길이: {hrir_l.shape[1]}, fs: {fs_hrtf} Hz")
        return hrir_l, hrir_r, azimuths, elevations, fs_hrtf
        
    except ImportError:
        pass
    except Exception as e:
        print(f"[SOFA] netCDF4 시도 실패: {e}")
    
    # 방법 2: h5py (일부 SOFA 파일은 HDF5로 직접 읽을 수 있음)
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            ir_data = np.array(f['Data.IR'])
            source_pos = np.array(f['SourcePosition'])
            fs_hrtf = float(np.array(f['Data.SamplingRate']).flat[0])
        
        hrir_l = ir_data[:, 0, :]
        hrir_r = ir_data[:, 1, :]
        azimuths = source_pos[:, 0]
        elevations = source_pos[:, 1]
        
        print(f"[SOFA] h5py로 읽기 성공")
        print(f"  방향 수: {len(azimuths)}, HRIR 길이: {hrir_l.shape[1]}, fs: {fs_hrtf} Hz")
        return hrir_l, hrir_r, azimuths, elevations, fs_hrtf
        
    except ImportError:
        pass
    except Exception as e:
        print(f"[SOFA] h5py 시도 실패: {e}")
    
    # 방법 3: scipy.io.netcdf (netCDF3 형식인 경우만 가능)
    try:
        from scipy.io import netcdf_file
        ds = netcdf_file(filepath, 'r', mmap=False)
        
        ir_data = np.array(ds.variables['Data.IR'].data)
        source_pos = np.array(ds.variables['SourcePosition'].data)
        fs_hrtf = float(np.array(ds.variables['Data.SamplingRate'].data).flat[0])
        
        hrir_l = ir_data[:, 0, :]
        hrir_r = ir_data[:, 1, :]
        azimuths = source_pos[:, 0]
        elevations = source_pos[:, 1]
        
        ds.close()
        print(f"[SOFA] scipy.io.netcdf로 읽기 성공")
        print(f"  방향 수: {len(azimuths)}, HRIR 길이: {hrir_l.shape[1]}, fs: {fs_hrtf} Hz")
        return hrir_l, hrir_r, azimuths, elevations, fs_hrtf
        
    except Exception as e:
        print(f"[SOFA] scipy.io.netcdf 시도 실패: {e}")
    
    raise RuntimeError(
        "SOFA 파일을 읽을 수 없습니다. "
        "netCDF4, h5py, 또는 scipy 중 하나가 필요합니다.\n"
        "설치: pip install netCDF4  또는  pip install h5py"
    )


# ============================================================
# 2. Cross-Correlation 함수 (매트랩 crossCorr 재현)
# ============================================================

def cross_corr(sig1, sig2):
    """
    두 신호의 교차 상관(cross-correlation)을 계산합니다.
    매트랩의 xcorr와 동일한 기능이며, 제공된 crossCorr 함수를 재현합니다.
    
    Parameters
    ----------
    sig1 : ndarray, shape (N,)
    sig2 : ndarray, shape (N,)
    
    Returns
    -------
    corr : ndarray, shape (2*N-1,)
        교차 상관 결과
    lags : ndarray, shape (2*N-1,)
        각 상관값에 대응하는 lag (샘플 단위)
    """
    N = len(sig1)
    # scipy의 fftconvolve를 이용한 효율적인 교차 상관
    # xcorr(sig1, sig2) = conv(sig1, flip(conj(sig2)))
    corr = fftconvolve(sig1, np.conj(sig2[::-1]), mode='full')
    lags = np.arange(-(N - 1), N)
    return corr, lags


def cross_corr_normalized(sig1, sig2):
    """
    정규화된 교차 상관 (lag=0에서의 상관 계수).
    매트랩의 xcorr(sig1, sig2, 0, 'coeff')와 동일합니다.
    
    Parameters
    ----------
    sig1 : ndarray, shape (N,)
    sig2 : ndarray, shape (N,)
    
    Returns
    -------
    coeff : float
        정규화된 상관 계수 (-1 ~ 1)
    """
    num = np.real(np.sum(sig1 * np.conj(sig2)))
    denom = np.sqrt(np.sum(np.abs(sig1) ** 2) * np.sum(np.abs(sig2) ** 2))
    if denom < 1e-12:
        return 0.0
    return num / denom


# ============================================================
# 3. Cross-Channel SSL 알고리즘
# ============================================================

def _corr_vectorized(P1L_rs, P1R_rs, HRTF_L, HRTF_R, xp):
    """
    모든 방향에 대한 normalized cross-correlation을 한 번에 계산합니다.

    FFT 결과(P1L_rs, P1R_rs)는 항상 numpy array로 전달됩니다.
    HRTF_L/R은 xp(numpy 또는 cupy) array입니다.
    GPU 사용 시 P1L_rs를 VRAM으로 전송 → 행렬 연산 → 결과를 RAM으로 반환.
    """
    if xp is not np:
        # GPU: spectrum 벡터만 전송 (HRTF는 이미 VRAM에 있음)
        P1L_g = xp.array(P1L_rs)
        P1R_g = xp.array(P1R_rs)
    else:
        P1L_g, P1R_g = P1L_rs, P1R_rs

    sig1 = P1L_g[None, :] * HRTF_R          # (N_dir, N_freq)
    sig2 = P1R_g[None, :] * HRTF_L
    num   = xp.real(xp.sum(sig1 * xp.conj(sig2), axis=1))
    denom = xp.sqrt(
        xp.sum(xp.abs(sig1) ** 2, axis=1) *
        xp.sum(xp.abs(sig2) ** 2, axis=1)
    )
    result = xp.where(denom > 1e-12, num / denom, 0.0)
    return result.get() if hasattr(result, 'get') else result


def _compute_heatmap_for_window(
    window_L, window_R, fs_audio,
    hrir_l, hrir_r, azimuths, elevations,
    HRTF_L, HRTF_R, NFFT,
    domain='frequency',
    rms_threshold_db=-70,
    frame_duration=0.04,
    overlap_ratio=0.5,
    xp=np
):
    """
    단일 500ms 윈도우에 대해 Cross-Channel SSL을 수행하고
    (N_dir,) corr_map을 반환합니다. (내부 함수)
    xp : numpy (CPU) 또는 cupy (GPU)
    """
    frame_length = NFFT
    shift_length = int(frame_length * (1 - overlap_ratio))
    hann_win     = np.hanning(frame_length)
    N_dir        = len(azimuths)

    corr_accum        = np.zeros(N_dir, dtype=np.float64)
    valid_frame_count = 0
    n_frames = max(1, (len(window_L) - frame_length) // shift_length + 1)

    for i in range(n_frames):
        start = i * shift_length
        end   = start + frame_length
        if end > len(window_L):
            break

        frame_L = window_L[start:end] * hann_win
        frame_R = window_R[start:end] * hann_win

        rms_L  = np.sqrt(np.mean(frame_L ** 2))
        rms_R  = np.sqrt(np.mean(frame_R ** 2))
        rms_db = 20 * np.log10(max(float(rms_L), float(rms_R)) + 1e-12)
        if rms_db < rms_threshold_db:
            continue

        valid_frame_count += 1

        if domain == 'frequency':
            # FFT는 항상 CPU(numpy)로 수행 — cuFFT 라이브러리 불필요
            frame_L_np = frame_L if xp is np else frame_L.get() if hasattr(frame_L, 'get') else np.asarray(frame_L)
            frame_R_np = frame_R if xp is np else frame_R.get() if hasattr(frame_R, 'get') else np.asarray(frame_R)
            FL     = np.fft.rfft(frame_L_np, n=NFFT)
            FR     = np.fft.rfft(frame_R_np, n=NFFT)
            N_freq = NFFT // 2 + 1
            P1L    = FL / NFFT;  P1L[1:-1] *= 2
            P1R    = FR / NFFT;  P1R[1:-1] *= 2

            N_hrtf_freq = HRTF_L.shape[1]
            if N_freq != N_hrtf_freq:
                ratio   = (N_freq - 1) / (N_hrtf_freq - 1)
                indices = np.clip(
                    np.round(np.arange(N_hrtf_freq) * ratio).astype(int),
                    0, N_freq - 1)
                P1L_rs = P1L[indices]
                P1R_rs = P1R[indices]
            else:
                P1L_rs, P1R_rs = P1L, P1R

            # 방향별 상관 계산: GPU면 HRTF와 함께 VRAM에서 처리
            corr_accum += _corr_vectorized(P1L_rs, P1R_rs, HRTF_L, HRTF_R, xp)

        else:  # time domain (GPU 미지원, CPU fallback)
            for j in range(N_dir):
                sig1 = fftconvolve(
                    np.array(frame_L), hrir_r[j, :], mode='full')
                sig2 = fftconvolve(
                    np.array(frame_R), hrir_l[j, :], mode='full')
                n    = min(len(sig1), len(sig2))
                corr_accum[j] += cross_corr_normalized(sig1[:n], sig2[:n])

    if valid_frame_count > 0:
        result = corr_accum / valid_frame_count
    else:
        result = corr_accum

    return result


def ssl_cross_channel(
    audio_L, audio_R, fs_audio,
    hrir_l, hrir_r, azimuths, elevations, fs_hrtf,
    frame_duration=0.04,
    overlap_ratio=0.5,
    domain='frequency',
    rms_threshold_db=-70,
    win_size=0.5,
    hop_size=0.1,
    device='cpu'
):
    """
    Cross-channel 기법을 이용한 음원 방향 추정.

    500ms 윈도우 / 100ms hop으로 시간축을 슬라이딩하며
    각 윈도우마다 히트맵을 계산합니다.

    Parameters
    ----------
    audio_L, audio_R : ndarray  — 좌/우 채널
    fs_audio : int              — 오디오 샘플링 레이트
    hrir_l, hrir_r : ndarray    — HRIR (N_dir, N_ir)
    azimuths, elevations : ndarray
    fs_hrtf : float
    frame_duration : float      — 내부 FFT 프레임 길이 (초)
    overlap_ratio : float       — 내부 프레임 오버랩
    domain : str                — 'frequency' 또는 'time'
    rms_threshold_db : float    — 무음 스킵 임계값
    win_size : float            — 슬라이딩 윈도우 크기 (초, 기본 0.5)
    hop_size : float            — 슬라이딩 hop 크기  (초, 기본 0.1)

    Returns
    -------
    result : dict
        'frames'          : list of dict — 각 윈도우의 결과
            {time_center, heatmap, corr_map, best_azimuth, best_elevation}
        'unique_azimuths' : ndarray
        'unique_elevations': ndarray
        'heatmap'         : ndarray — 전체 평균 히트맵 (N_el, N_az)
        'best_azimuth'    : float
        'best_elevation'  : float
    """
    NFFT = int(2 ** np.round(np.log2(fs_audio * frame_duration)))

    # ── 연산 백엔드 결정 (GPU / CPU) ──
    xp = np  # 기본: numpy (CPU)
    if device == 'gpu':
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            xp = cp
            print(f"[device] GPU 사용: {cp.cuda.Device(0)}")
        except Exception as e:
            print(f"[device] GPU 초기화 실패 ({e}) → CPU로 fallback")

    print(f"\n[SSL] 파라미터:")
    print(f"  슬라이딩 윈도우: {win_size*1000:.0f}ms / hop {hop_size*1000:.0f}ms")
    print(f"  내부 FFT 프레임: {NFFT} samples ({NFFT/fs_audio*1000:.1f}ms)")
    print(f"  HRTF 방향 수: {len(azimuths)},  도메인: {domain},  device: {device}")

    N_dir    = len(azimuths)
    win_samp = int(win_size * fs_audio)
    hop_samp = int(hop_size * fs_audio)

    HRTF_L = HRTF_R = None
    if domain == 'frequency':
        HRTF_L_np = np.fft.rfft(hrir_l, n=NFFT, axis=1)
        HRTF_R_np = np.fft.rfft(hrir_r, n=NFFT, axis=1)
        # GPU면 VRAM에 미리 올려두기 (매 윈도우마다 복사 없이 재사용)
        HRTF_L = xp.array(HRTF_L_np) if xp is not np else HRTF_L_np
        HRTF_R = xp.array(HRTF_R_np) if xp is not np else HRTF_R_np

    # ── 히트맵 구성용 인덱스 ──
    unique_az = np.sort(np.unique(azimuths))
    unique_el = np.sort(np.unique(elevations))
    az_to_idx = {az: i for i, az in enumerate(unique_az)}
    el_to_idx = {el: i for i, el in enumerate(unique_el)}

    def corr_to_heatmap(corr_map):
        hm = np.full((len(unique_el), len(unique_az)), np.nan)
        for d in range(N_dir):
            ai = az_to_idx.get(azimuths[d])
            ei = el_to_idx.get(elevations[d])
            if ai is not None and ei is not None:
                hm[ei, ai] = corr_map[d]
        return hm

    # ── 슬라이딩 윈도우 루프 ──
    n_wins   = max(1, (len(audio_L) - win_samp) // hop_samp + 1)
    frames   = []
    accum_all = np.zeros(N_dir)

    print(f"  총 윈도우 수: {n_wins}")
    for wi in range(n_wins):
        s = wi * hop_samp
        e = s + win_samp
        if e > len(audio_L):
            break

        win_L = audio_L[s:e]
        win_R = audio_R[s:e]

        corr_map = _compute_heatmap_for_window(
            win_L, win_R, fs_audio,
            hrir_l, hrir_r, azimuths, elevations,
            HRTF_L, HRTF_R, NFFT,
            domain=domain,
            rms_threshold_db=rms_threshold_db,
            frame_duration=frame_duration,
            overlap_ratio=overlap_ratio,
            xp=xp
        )

        best_idx = int(np.argmax(corr_map))
        hm = corr_to_heatmap(corr_map)
        accum_all += corr_map

        frames.append({
            'win_idx'       : wi,
            'time_start'    : s / fs_audio,
            'time_end'      : e / fs_audio,
            'time_center'   : (s + e) / 2 / fs_audio,
            'corr_map'      : corr_map,
            'heatmap'       : hm,
            'best_azimuth'  : float(azimuths[best_idx]),
            'best_elevation': float(elevations[best_idx]),
            'best_corr'     : float(corr_map[best_idx])
        })

        if (wi + 1) % 20 == 0 or (wi + 1) == n_wins:
            print(f"  처리 중... {wi+1}/{n_wins} 윈도우", end='\r')

    print(f"\n  완료: {len(frames)} 윈도우")

    # ── 전체 평균 히트맵 ──
    mean_corr = accum_all / max(len(frames), 1)
    mean_hm   = corr_to_heatmap(mean_corr)
    best_all  = int(np.argmax(mean_corr))

    return {
        'frames'           : frames,
        'unique_azimuths'  : unique_az,
        'unique_elevations': unique_el,
        'heatmap'          : mean_hm,
        'corr_map'         : mean_corr,
        'best_azimuth'     : float(azimuths[best_all]),
        'best_elevation'   : float(elevations[best_all]),
        'best_corr'        : float(mean_corr[best_all])
    }


# ============================================================
# 4. 시각화
# ============================================================

def plot_heatmap(result, save_path=None):
    """
    방위각-고도각 히트맵을 그립니다. (전체 평균 — 단일 이미지)

    Parameters
    ----------
    result : dict
        ssl_cross_channel()의 반환값
    save_path : str, optional
        저장 경로 (None이면 plt.show())
    """
    heatmap   = result['heatmap']
    unique_az = result['unique_azimuths']
    unique_el = result['unique_elevations']
    best_az   = result['best_azimuth']
    best_el   = result['best_elevation']

    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(
        heatmap,
        aspect='auto', origin='lower', cmap='hot', interpolation='nearest',
        extent=[unique_az[0], unique_az[-1], unique_el[0], unique_el[-1]]
    )
    ax.plot(best_az, best_el, 'c*', markersize=20, markeredgecolor='white',
            markeredgewidth=1.5, label=f'Peak: az={best_az:.0f}°, el={best_el:.0f}°')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Cross-Correlation Coefficient', fontsize=11)
    ax.set_xlabel('Azimuth (°)', fontsize=13)
    ax.set_ylabel('Elevation (°)', fontsize=13)
    ax.set_title('Sound Source Localization — Cross-Channel HRTF Method', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, color='white')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장] 히트맵 저장: {save_path}")
    else:
        plt.show()

    return fig


# ============================================================
# 5-b. MP4 저장 (프레임별 히트맵 + 오디오)
# ============================================================

def save_video(result, audio_L, audio_R, fs_audio,
               output_path, gt_path=None,
               fps=None, dpi=100):
    """
    슬라이딩 윈도우 결과를 MP4 영상으로 저장합니다.
    - 각 프레임 = 하나의 500ms 윈도우의 히트맵
    - 오디오 트랙이 병합됩니다 (ffmpeg 필요)
    - Ground Truth JSON이 주어지면 실제 음원 위치를 오버레이합니다.

    Parameters
    ----------
    result : dict   ssl_cross_channel() 반환값
    audio_L, audio_R : ndarray   원본 오디오
    fs_audio : int
    output_path : str   최종 MP4 저장 경로
    gt_path : str, optional   Ground Truth JSON 경로
    fps : float, optional   영상 FPS (None이면 1/hop_size 자동 계산)
    dpi : int   렌더링 해상도
    """
    frames    = result['frames']
    unique_az = result['unique_azimuths']
    unique_el = result['unique_elevations']
    n_frames  = len(frames)

    if n_frames == 0:
        print("[경고] 프레임이 없어 영상을 저장하지 않습니다.")
        return

    # hop 크기로 FPS 추정
    if fps is None:
        if n_frames > 1:
            hop = frames[1]['time_start'] - frames[0]['time_start']
            fps = 1.0 / hop if hop > 0 else 10.0
        else:
            fps = 10.0
    fps = float(fps)

    # ── Ground Truth 로드 ──
    gt_events = []
    if gt_path and os.path.isfile(gt_path):
        import json as _json
        with open(gt_path, 'r', encoding='utf-8') as fp:
            gt = _json.load(fp)
        gt_events = gt.get('events', [])
        print(f"[GT] {len(gt_events)}개 이벤트 오버레이")

    # 컬러맵 범위 (전체 프레임 기준)
    all_vals = np.concatenate([f['corr_map'] for f in frames])
    vmin = np.nanpercentile(all_vals, 5)
    vmax = np.nanpercentile(all_vals, 99)

    single_el = (len(unique_el) == 1)  # HRTF에 elevation이 1개뿐인 경우

    # ── Figure 구성 ──
    fig = plt.figure(figsize=(14, 7))
    if single_el:
        # elevation 1개: 상단(azimuth 막대그래프) + 하단(시간 진행바)
        gs   = GridSpec(2, 1, figure=fig, height_ratios=[5, 1], hspace=0.4)
        ax_hm  = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        # 막대 초기화
        corr0  = frames[0]['corr_map']
        bars   = ax_hm.bar(unique_az, corr0, width=0.8, color='#ff6600',
                           edgecolor='none', align='center')
        peak_line = ax_hm.axvline(x=unique_az[0], color='cyan', linewidth=2,
                                  linestyle='--', label='Peak')
        gt_lines  = []   # GT용 axvline 핸들 목록 (동적으로 관리)

        ax_hm.set_xlim(unique_az[0] - 1, unique_az[-1] + 1)
        ax_hm.set_ylim(vmin - 0.02, vmax + 0.02)
        ax_hm.set_xlabel('Azimuth (°)', fontsize=12)
        ax_hm.set_ylabel('Cross-Correlation', fontsize=12)
        ax_hm.text(0.99, 0.97,
                   f'(HRTF has only elevation={unique_el[0]:.0f}°)',
                   transform=ax_hm.transAxes, ha='right', va='top',
                   fontsize=9, color='gray')
        ax_hm.grid(True, alpha=0.3, axis='y')
        ax_hm.legend(fontsize=10, loc='upper left')
        title = ax_hm.set_title('', fontsize=13)

    else:
        # 2D 히트맵
        gs   = GridSpec(2, 1, figure=fig, height_ratios=[5, 1], hspace=0.35)
        ax_hm  = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        dummy_hm = frames[0]['heatmap']
        extent   = [unique_az[0], unique_az[-1], unique_el[0], unique_el[-1]]
        im = ax_hm.imshow(
            dummy_hm, aspect='auto', origin='lower', cmap='hot',
            interpolation='bilinear', extent=extent, vmin=vmin, vmax=vmax
        )
        cbar = fig.colorbar(im, ax=ax_hm, shrink=0.85, pad=0.01)
        cbar.set_label('Normalized Cross-Correlation', fontsize=10)

        peak_marker, = ax_hm.plot([], [], 'c*', markersize=16,
                                  markeredgecolor='white', markeredgewidth=1.5)
        gt_markers,  = ax_hm.plot([], [], 'g^', markersize=12,
                                  markeredgecolor='white', markeredgewidth=1,
                                  label='Ground Truth')

        ax_hm.set_xlabel('Azimuth (°)', fontsize=12)
        ax_hm.set_ylabel('Elevation (°)', fontsize=12)
        ax_hm.grid(True, alpha=0.2, color='white')
        title = ax_hm.set_title('', fontsize=13)

    # ── 시간 진행바 (공통) ──
    total_dur = len(audio_L) / fs_audio
    ax_bar.barh(0, total_dur, color='#444', height=0.4, align='center')
    bar_fg = ax_bar.barh(0, 0, color='#00aaff', height=0.4, align='center')
    ax_bar.set_xlim(0, total_dur)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.set_xlabel('Time (s)', fontsize=10)
    ax_bar.set_yticks([])
    time_txt = ax_bar.text(0.01, 0, '', va='center', ha='left',
                           fontsize=9, color='white',
                           transform=ax_bar.get_yaxis_transform())

    def _active_gt(t_center):
        return [ev for ev in gt_events
                if ev['start_time'] <= t_center <= ev['end_time']]

    def update(fi):
        fr  = frames[fi]
        baz = fr['best_azimuth']
        bel = fr['best_elevation']
        tc  = fr['time_center']
        ts  = fr['time_start']
        te  = fr['time_end']
        active = _active_gt(tc)

        title.set_text(
            f'SSL  [{ts:.2f}s ~ {te:.2f}s]  '
            f'Peak: az={baz:.0f}°  el={bel:.0f}°'
            + (f'  |  GT active: {len(active)}' if active else '')
        )
        bar_fg[0].set_width(tc)
        time_txt.set_text(f'{tc:.2f}s')

        if single_el:
            # 막대 높이 갱신
            corr = fr['corr_map']
            for bar, h in zip(bars, corr):
                bar.set_height(h)
                # 피크 막대 색 강조
                bar.set_color('#ff6600')
            peak_idx = int(np.argmax(corr))
            bars[peak_idx].set_color('cyan')
            peak_line.set_xdata([baz])

            # GT 수직선 갱신 (기존 것 제거 후 재생성)
            for ln in gt_lines:
                ln.remove()
            gt_lines.clear()
            for ev in active:
                ln = ax_hm.axvline(x=ev['azimuth'], color='lime',
                                   linewidth=2, alpha=0.8, linestyle=':')
                gt_lines.append(ln)

            return (*bars, peak_line, bar_fg[0], time_txt, title)

        else:
            # 2D 히트맵 갱신
            im.set_data(fr['heatmap'])
            im.set_clim(vmin, vmax)
            peak_marker.set_data([baz], [bel])

            if active:
                gx = [ev['azimuth']   for ev in active]
                gy = [ev['elevation'] for ev in active]
                gt_markers.set_data(gx, gy)
            else:
                gt_markers.set_data([], [])

            return im, peak_marker, gt_markers, bar_fg[0], time_txt, title

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000/fps, blit=True
    )

    # ── 영상 임시 저장 (소리 없음) ──
    tmp_video = output_path.replace('.mp4', '_noaudio.mp4')

    ffmpeg_bin = _find_ffmpeg()
    writer = animation.FFMpegWriter(
        fps=fps, metadata={'title': 'SSL Heatmap'},
        extra_args=['-vcodec', 'libopenh264', '-pix_fmt', 'yuv420p']
    )
    # FFMpegWriter가 찾을 수 있도록 PATH 설정
    old_path = os.environ.get('PATH', '')
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + old_path

    print(f"[영상] {n_frames}프레임 렌더링 중... (fps={fps:.1f})")
    ani.save(tmp_video, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"[영상] 렌더링 완료: {tmp_video}")

    # ── 오디오 WAV 임시 저장 ──
    import soundfile as sf
    tmp_audio = output_path.replace('.mp4', '_audio.wav')
    stereo = np.stack([audio_L, audio_R], axis=1).astype(np.float32)
    sf.write(tmp_audio, stereo, fs_audio)

    # ── ffmpeg로 오디오 병합 ──
    cmd = [
        ffmpeg_bin, '-y',
        '-i', tmp_video,
        '-i', tmp_audio,
        '-c:v', 'copy',
        '-c:a', 'libmp3lame', '-b:a', '192k',
        '-shortest',
        output_path
    ]
    print(f"[영상] 오디오 병합 중...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # 임시 파일 삭제
    os.remove(tmp_video)
    os.remove(tmp_audio)
    os.environ['PATH'] = old_path

    print(f"[저장] MP4 저장 완료: {output_path}")
    return output_path


def _find_ffmpeg():
    """시스템 또는 anaconda에서 ffmpeg 바이너리를 찾습니다."""
    candidates = ['ffmpeg']
    home = os.path.expanduser('~')
    for base in [
        os.path.join(home, 'anaconda3', 'bin'),
        os.path.join(home, 'miniconda3', 'bin'),
        '/usr/local/bin', '/usr/bin',
    ]:
        candidates.append(os.path.join(base, 'ffmpeg'))

    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # 마지막 수단: which
    result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise FileNotFoundError(
        "ffmpeg를 찾을 수 없습니다. conda install ffmpeg 또는 apt install ffmpeg"
    )


# ============================================================
# 5-c. 오디오 재생 (sounddevice)
# ============================================================

def play_audio(audio_L, audio_R, fs):
    """바이노럴 오디오를 백그라운드 스레드로 재생합니다."""
    try:
        import sounddevice as sd
    except ImportError:
        print("[재생] sounddevice 미설치 — 재생 건너뜀")
        return None

    stereo = np.stack([audio_L, audio_R], axis=1).astype(np.float32)

    def _play():
        sd.play(stereo, samplerate=fs, blocking=True)

    t = threading.Thread(target=_play, daemon=True)
    t.start()
    print(f"[재생] 오디오 재생 시작 ({len(audio_L)/fs:.1f}초)")
    return t


def plot_frame_trajectory(result, save_path=None):
    """
    시간에 따른 윈도우별 추정 방향 변화를 그립니다.

    Parameters
    ----------
    result : dict
        ssl_cross_channel()의 반환값
    save_path : str, optional
        저장 경로
    """
    frames = result.get('frames', [])
    if not frames:
        print("[경고] 프레임 결과가 없습니다.")
        return

    times  = [fr['time_center']    for fr in frames]
    az_est = [fr['best_azimuth']   for fr in frames]
    el_est = [fr['best_elevation'] for fr in frames]
    corrs  = [fr['best_corr']      for fr in frames]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(times, az_est, 'b.-', markersize=3, linewidth=0.8)
    axes[0].set_ylabel('Azimuth (°)', fontsize=11)
    axes[0].set_title('Frame-by-Frame Estimated Direction', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times, el_est, 'r.-', markersize=3, linewidth=0.8)
    axes[1].set_ylabel('Elevation (°)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times, corrs, 'g.-', markersize=3, linewidth=0.8)
    axes[2].set_ylabel('Correlation Coeff.', fontsize=11)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장] 궤적 그래프 저장: {save_path}")
    else:
        plt.show()
    
    return fig


# ============================================================
# 5. 데모 데이터 생성 (테스트용)
# ============================================================

def generate_demo_data(
    target_azimuth=30.0,
    target_elevation=0.0,
    duration=1.0,
    fs=48000,
    n_az=24,
    n_el=13,
    hrir_length=128
):
    """
    테스트용 데모 데이터를 생성합니다.
    간단한 구형 머리 모델 기반의 HRIR과 바이노럴 오디오를 만듭니다.
    
    Parameters
    ----------
    target_azimuth : float
        실제 음원 방위각 (도)
    target_elevation : float
        실제 음원 고도각 (도)
    duration : float
        오디오 길이 (초)
    fs : int
        샘플링 레이트
    n_az : int
        방위각 개수 (0° ~ 345°, 15° 간격이면 24)
    n_el : int
        고도각 개수
    hrir_length : int
        HRIR 길이 (샘플)
    
    Returns
    -------
    demo : dict
        모든 데모 데이터를 담은 딕셔너리
    """
    print(f"\n[데모] 테스트 데이터 생성 중...")
    print(f"  목표 방향: azimuth={target_azimuth}°, elevation={target_elevation}°")
    
    head_radius = 0.0875  # 머리 반지름 (m)
    speed_of_sound = 343.0  # 음속 (m/s)
    
    # 방위각, 고도각 그리드
    az_values = np.linspace(0, 345, n_az)
    el_values = np.linspace(-45, 90, n_el)
    
    az_grid, el_grid = np.meshgrid(az_values, el_values)
    azimuths = az_grid.flatten()
    elevations = el_grid.flatten()
    N_dir = len(azimuths)
    
    # 간단한 구형 머리 모델 HRIR 생성
    hrir_l = np.zeros((N_dir, hrir_length))
    hrir_r = np.zeros((N_dir, hrir_length))
    
    for d in range(N_dir):
        az_rad = np.deg2rad(azimuths[d])
        el_rad = np.deg2rad(elevations[d])
        
        # ITD 계산 (Woodworth 모델)
        itd = (head_radius / speed_of_sound) * (np.sin(az_rad) * np.cos(el_rad) 
               + az_rad * np.cos(el_rad))
        # 주의: 간단한 근사
        itd = (head_radius / speed_of_sound) * np.sin(az_rad) * np.cos(el_rad)
        
        delay_samples_L = int(round(max(0, -itd) * fs))
        delay_samples_R = int(round(max(0, itd) * fs))
        
        # ILD 계산 (간단한 모델)
        ild_db = 10 * np.sin(az_rad) * np.cos(el_rad)
        gain_L = 10 ** (-ild_db / 20)
        gain_R = 10 ** (ild_db / 20)
        
        # 간단한 임펄스 (가우시안 펄스)
        t_ir = np.arange(hrir_length) / fs
        pulse_width = 0.0005  # 0.5ms
        
        if delay_samples_L < hrir_length:
            center_L = delay_samples_L
            hrir_l[d, :] = gain_L * np.exp(-((t_ir - center_L / fs) ** 2) 
                                            / (2 * pulse_width ** 2))
        
        if delay_samples_R < hrir_length:
            center_R = delay_samples_R
            hrir_r[d, :] = gain_R * np.exp(-((t_ir - center_R / fs) ** 2) 
                                            / (2 * pulse_width ** 2))
    
    # --- 바이노럴 오디오 생성 ---
    # 원본 신호: 백색 잡음
    n_samples = int(duration * fs)
    source = np.random.randn(n_samples) * 0.5
    
    # 목표 방향에 가장 가까운 HRIR 찾기
    az_diff = np.abs(azimuths - target_azimuth)
    el_diff = np.abs(elevations - target_elevation)
    target_idx = np.argmin(az_diff + el_diff)
    
    actual_az = azimuths[target_idx]
    actual_el = elevations[target_idx]
    print(f"  실제 적용된 방향: azimuth={actual_az}°, elevation={actual_el}°")
    
    # 컨볼루션으로 바이노럴 신호 생성
    audio_L = fftconvolve(source, hrir_l[target_idx, :], mode='full')[:n_samples]
    audio_R = fftconvolve(source, hrir_r[target_idx, :], mode='full')[:n_samples]
    
    # 약간의 잡음 추가
    noise_level = 0.01
    audio_L += noise_level * np.random.randn(n_samples)
    audio_R += noise_level * np.random.randn(n_samples)
    
    return {
        'audio_L': audio_L,
        'audio_R': audio_R,
        'fs': fs,
        'hrir_l': hrir_l,
        'hrir_r': hrir_r,
        'azimuths': azimuths,
        'elevations': elevations,
        'fs_hrtf': float(fs),
        'target_azimuth': actual_az,
        'target_elevation': actual_el
    }


# ============================================================
# 6. 메인
# ============================================================

def _load_audio(audio_path):
    """WAV 파일을 읽어 float64 L/R 채널과 fs를 반환합니다."""
    fs_audio, audio_data = wavfile.read(audio_path)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float64) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float64) / 2147483648.0
    elif audio_data.dtype == np.float32:
        audio_data = audio_data.astype(np.float64)
    if audio_data.ndim == 1:
        raise ValueError("모노 오디오입니다. 스테레오(바이노럴) WAV 파일이 필요합니다.")
    return audio_data[:, 0], audio_data[:, 1], int(fs_audio)


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Channel HRTF 기반 음원 방향 추정 (SSL) — MP4 출력',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # SOFA + WAV → MP4 저장 (기본)
  python heatmap.py --sofa hrtf/KEMAR_HRTF.sofa --audio output/scene_01.wav

  # Ground Truth 오버레이 포함
  python heatmap.py --sofa hrtf/KEMAR_HRTF.sofa --audio output/scene_01.wav \\
                    --gt output/scene_01_gt.json

  # 오디오 재생도 함께
  python heatmap.py --sofa hrtf/KEMAR_HRTF.sofa --audio output/scene_01.wav --play

  # 데모 모드 (합성 데이터 자동 생성)
  python heatmap.py --demo --target-az 45

  # 출력 MP4 경로 지정
  python heatmap.py --sofa hrtf/KEMAR_HRTF.sofa --audio output/scene_01.wav \\
                    --output result.mp4
        """
    )

    # 입력
    parser.add_argument('--sofa',  type=str, help='HRTF SOFA 파일 경로')
    parser.add_argument('--audio', type=str, help='바이노럴 스테레오 WAV 파일 경로')
    parser.add_argument('--gt',    type=str, default=None,
                        help='Ground Truth JSON 경로 (선택)')
    parser.add_argument('--demo',       action='store_true',
                        help='데모 모드 (SOFA 없이 내장 모델로 테스트)')
    parser.add_argument('--target-az',  type=float, default=45.0,
                        help='데모 목표 방위각 (도)')
    parser.add_argument('--target-el',  type=float, default=0.0,
                        help='데모 목표 고도각 (도)')

    # SSL 파라미터
    parser.add_argument('--win-size',       type=float, default=0.5,
                        help='슬라이딩 윈도우 크기 (초, 기본 0.5)')
    parser.add_argument('--hop-size',       type=float, default=0.1,
                        help='슬라이딩 hop 크기 (초, 기본 0.1)')
    parser.add_argument('--frame-duration', type=float, default=0.04,
                        help='내부 FFT 프레임 길이 (초, 기본 0.04)')
    parser.add_argument('--domain', type=str, default='frequency',
                        choices=['frequency', 'time'], help='처리 도메인')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='연산 장치: cpu (기본) 또는 gpu (CuPy 필요)')

    # 출력
    parser.add_argument('--output', type=str, default=None,
                        help='출력 MP4 파일 경로 (기본: output/ssl_<audio_stem>.mp4)')
    parser.add_argument('--play',   action='store_true',
                        help='MP4 저장 전에 오디오를 재생합니다')
    parser.add_argument('--no-video', action='store_true',
                        help='MP4 저장 없이 전체 평균 히트맵 PNG만 저장')
    parser.add_argument('--dpi',    type=int, default=100,
                        help='렌더링 DPI (기본 100)')

    args = parser.parse_args()

    # ── 데이터 로드 ──────────────────────────────────────────
    if args.demo:
        print("=" * 60)
        print("  Cross-Channel SSL — 데모 모드")
        print("=" * 60)
        demo = generate_demo_data(
            target_azimuth=args.target_az,
            target_elevation=args.target_el
        )
        audio_L    = demo['audio_L']
        audio_R    = demo['audio_R']
        fs_audio   = demo['fs']
        hrir_l     = demo['hrir_l']
        hrir_r     = demo['hrir_r']
        azimuths   = demo['azimuths']
        elevations = demo['elevations']
        fs_hrtf    = demo['fs_hrtf']
        audio_stem = 'demo'

    elif args.sofa and args.audio:
        print("=" * 60)
        print("  Cross-Channel SSL — 파일 모드")
        print("=" * 60)
        hrir_l, hrir_r, azimuths, elevations, fs_hrtf = read_sofa(args.sofa)
        audio_L, audio_R, fs_audio = _load_audio(args.audio)
        audio_stem = os.path.splitext(os.path.basename(args.audio))[0]

        print(f"[오디오] {args.audio}")
        print(f"  길이: {len(audio_L)/fs_audio:.2f}초, fs: {fs_audio} Hz")
        if abs(fs_audio - fs_hrtf) > 1:
            print(f"[경고] 오디오 fs({fs_audio})와 HRTF fs({fs_hrtf})가 다릅니다.")

    else:
        parser.print_help()
        print("\n[오류] --demo 또는 (--sofa + --audio) 조합이 필요합니다.")
        sys.exit(1)

    # ── GT JSON 자동 감지 ─────────────────────────────────────
    gt_path = args.gt
    if gt_path is None and args.audio:
        auto_gt = os.path.splitext(os.path.abspath(args.audio))[0] + '_gt.json'
        if os.path.isfile(auto_gt):
            gt_path = auto_gt
            print(f"[GT] 자동 감지: {auto_gt}")

    # ── 출력 경로 결정 ────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    w_ms = int(args.win_size * 1000)
    h_ms = int(args.hop_size * 1000)

    if args.output:
        video_path = args.output
    else:
        video_path = os.path.join(output_dir, f'ssl_{audio_stem}_w{w_ms}ms_h{h_ms}ms.mp4')

    # ── SSL 수행 ──────────────────────────────────────────────
    result = ssl_cross_channel(
        audio_L, audio_R, fs_audio,
        hrir_l, hrir_r, azimuths, elevations, fs_hrtf,
        frame_duration=args.frame_duration,
        domain=args.domain,
        win_size=args.win_size,
        hop_size=args.hop_size,
        device=args.device
    )

    # ── 오디오 재생 (옵션) ────────────────────────────────────
    play_thread = None
    if args.play:
        play_thread = play_audio(audio_L, audio_R, fs_audio)

    # ── 출력 ─────────────────────────────────────────────────
    if args.no_video:
        stem = f'ssl_{audio_stem}_w{w_ms}ms_h{h_ms}ms' if not args.output else os.path.splitext(args.output)[0]
        heatmap_path    = os.path.join(output_dir, f'{stem}_heatmap.png')
        trajectory_path = os.path.join(output_dir, f'{stem}_trajectory.png')
        plot_heatmap(result, save_path=heatmap_path)
        plot_frame_trajectory(result, save_path=trajectory_path)
    else:
        save_video(
            result, audio_L, audio_R, fs_audio,
            output_path=video_path,
            gt_path=gt_path,
            dpi=args.dpi
        )

    # ── 결과 요약 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  최종 결과 요약")
    print("=" * 60)
    print(f"  추정 방위각:  {result['best_azimuth']:.1f}°")
    print(f"  추정 고도각:  {result['best_elevation']:.1f}°")
    print(f"  최대 상관계수: {result['best_corr']:.4f}")
    if not args.no_video:
        print(f"  MP4 저장:     {video_path}")

    if args.demo:
        az_err = abs(result['best_azimuth'] - demo['target_azimuth'])
        el_err = abs(result['best_elevation'] - demo['target_elevation'])
        az_err = min(az_err, 360 - az_err)
        print(f"\n  [데모 검증]")
        print(f"  실제 방위각: {demo['target_azimuth']:.1f}° → 오차: {az_err:.1f}°")
        print(f"  실제 고도각: {demo['target_elevation']:.1f}° → 오차: {el_err:.1f}°")

    if play_thread is not None:
        play_thread.join()

    return result


if __name__ == '__main__':
    main()