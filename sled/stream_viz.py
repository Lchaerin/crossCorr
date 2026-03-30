#!/home/rllab/anaconda3/bin/python
"""
SLED v3 — Real-time Binaural Visualization
===========================================
마이크로폰(binaural 헤드셋 등)에서 실시간으로 들어오는 스테레오 오디오를
SLED 모델로 추론하여 DOA / class / confidence를 실시간으로 시각화합니다.

Usage
-----
    # 오디오 장치 목록 확인
    python sled/stream_viz.py --list-devices

    # 실행 (기본 입력 장치)
    python sled/stream_viz.py --ckpt ./checkpoints/sled_best.pt \\
        --sofa-path ./hrtf/p0001.sofa

    # 장치 지정 + 클래스맵
    python sled/stream_viz.py --ckpt ./checkpoints/sled_best.pt \\
        --sofa-path ./hrtf/p0001.sofa \\
        --class-map ./data/meta/class_map.json \\
        --audio-device 4 --conf-thresh 0.35

Architecture
------------
  Main thread      : matplotlib FuncAnimation (display)
  inference_thread : 모델 추론 루프 (~20 Hz)
  consumer_thread  : chunk_queue → ring buffer
  sounddevice cb   : indata.copy() → chunk_queue  (GIL 최소화)
"""

import argparse
import collections
import os
import sys
import threading
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import sounddevice as sd
except OSError as e:
    sys.exit(f'[ERROR] sounddevice 초기화 실패: {e}\n'
             '  → libportaudio.so 위치를 확인하거나 conda-forge portaudio를 설치하세요.')

import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sled.model.sled import SLEDv3

# ── 상수 ───────────────────────────────────────────────────────────────────────
HOP_SAMPLES   = 960
SAMPLE_RATE   = 48_000
FIG_W, FIG_H  = 1680, 700
DISPLAY_FPS   = 15          # 화면 갱신 빈도 (Hz)
INFER_HZ      = 20          # 목표 추론 빈도 (Hz)
BUFFER_SECS   = 12          # ring buffer 길이 (초)
TRAIL_LEN     = 12          # 궤적 표시 프레임 수

_SLOT_COLORS  = ['#e74c3c', '#2ecc71', '#3498db']   # slot 0,1,2


# =============================================================================
# Utilities
# =============================================================================

def doa_to_az_el(doa: np.ndarray):
    """SLED 단위벡터 [3] → (azimuth_deg CW, elevation_deg)."""
    x, y, z = doa
    el = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(y, x)) % 360.0)
    return az, el


def build_id_to_label(class_map_path: str | None) -> dict:
    if not class_map_path or not os.path.exists(class_map_path):
        return {}
    with open(class_map_path) as f:
        cm = json.load(f)
    id_to_label = {}
    for key, cid in cm.items():
        parts = key.split('/', 1)
        label = parts[0] if len(parts) == 2 else os.path.splitext(parts[0])[0]
        if cid not in id_to_label:
            id_to_label[cid] = label
    return id_to_label


# =============================================================================
# Ring buffer (thread-safe)
# =============================================================================

class AudioRingBuffer:
    """스테레오 오디오 ring buffer (circular pointer, np.roll 없음).

    write()는 포인터만 이동하므로 O(n_new)로 빠름.
    sounddevice 콜백 thread와 추론 thread가 공유.
    """

    def __init__(self, n_channels: int = 2, duration_sec: float = BUFFER_SECS):
        self.capacity   = int(SAMPLE_RATE * duration_sec)
        self.n_channels = n_channels
        self._buf       = np.zeros((n_channels, self.capacity), dtype=np.float32)
        self._lock      = threading.Lock()
        self._ptr       = 0   # 다음 쓰기 위치 (mod capacity)
        self._filled    = 0   # 유효 샘플 수 (최대 capacity)

    def write(self, data: np.ndarray):
        """data: [n_channels, n_samples]  or  [n_samples, n_channels]"""
        if data.ndim == 2 and data.shape[0] != self.n_channels:
            data = data.T
        n = data.shape[1]
        with self._lock:
            if n >= self.capacity:
                self._buf[:] = data[:, -self.capacity:]
                self._ptr    = 0
                self._filled = self.capacity
                return
            end = self._ptr + n
            if end <= self.capacity:
                self._buf[:, self._ptr:end] = data
            else:
                split = self.capacity - self._ptr
                self._buf[:, self._ptr:]         = data[:, :split]
                self._buf[:, :end - self.capacity] = data[:, split:]
            self._ptr    = end % self.capacity
            self._filled = min(self._filled + n, self.capacity)

    def _read_last_locked(self, n_samples: int) -> np.ndarray:
        """lock 보유 상태에서 마지막 n_samples 반환."""
        end   = self._ptr
        start = (end - n_samples) % self.capacity
        if start < end:
            return self._buf[:, start:end].copy()
        return np.concatenate([self._buf[:, start:], self._buf[:, :end]], axis=1)

    def read_last(self, n_samples: int) -> np.ndarray | None:
        """마지막 n_samples 반환. 부족하면 None."""
        with self._lock:
            if self._filled < n_samples:
                return None
            return self._read_last_locked(n_samples)

    def read_display(self, n_samples: int) -> np.ndarray:
        """표시용: 부족하면 앞을 0 패딩하여 반환 [2, n_samples]."""
        with self._lock:
            out    = np.zeros((self.n_channels, n_samples), dtype=np.float32)
            n_copy = min(self._filled, n_samples)
            if n_copy > 0:
                out[:, -n_copy:] = self._read_last_locked(n_copy)
            return out


# =============================================================================
# Prediction store (thread-safe)
# =============================================================================

class PredictionStore:
    """최신 추론 결과 + 궤적 기록 보관소."""

    def __init__(self, n_slots: int = 3, trail_len: int = TRAIL_LEN):
        self.trail_len = trail_len
        self._lock     = threading.Lock()
        # 현재 프레임 예측
        self.cls  = np.zeros(n_slots, dtype=np.int64)
        self.doa  = np.zeros((n_slots, 3), dtype=np.float32)
        self.doa[:, 0] = 1.0   # 초기: 정면
        self.conf = np.zeros(n_slots, dtype=np.float32)
        # 궤적 (deque of dicts)
        self.trail: collections.deque = collections.deque(maxlen=trail_len)
        # 메트릭
        self.infer_ms   = 0.0
        self.infer_time = 0.0
        self.ready      = False

    def update(self, cls, doa, conf, infer_ms: float):
        with self._lock:
            self.cls       = cls.copy()
            self.doa       = doa.copy()
            self.conf      = conf.copy()
            self.infer_ms  = infer_ms
            self.infer_time = time.time()
            self.trail.append({
                'doa' : doa.copy(),
                'conf': conf.copy(),
            })
            self.ready = True

    def snapshot(self):
        with self._lock:
            return (
                self.cls.copy(),
                self.doa.copy(),
                self.conf.copy(),
                list(self.trail),
                self.infer_ms,
                self.infer_time,
                self.ready,
            )


# =============================================================================
# Inference thread
# =============================================================================

def inference_worker(
    model: torch.nn.Module,
    audio_buf: AudioRingBuffer,
    pred_store: PredictionStore,
    window_samples: int,
    device: str,
    stop_event: threading.Event,
):
    """백그라운드 추론 루프."""
    interval = 1.0 / INFER_HZ
    model.eval()

    while not stop_event.is_set():
        t0   = time.perf_counter()
        data = audio_buf.read_last(window_samples)

        if data is not None:
            # 피크 정규화 (MRS 녹음과 동일한 방식)
            peak = np.abs(data).max()
            if peak > 1e-6:
                data = data / peak * 0.5

            tensor = torch.from_numpy(data).unsqueeze(0).to(device)  # [1, 2, N]
            with torch.no_grad():
                result = model(tensor, gt=None)

            pred = result['layer_preds'][-1]   # 마지막 decoder layer
            # 마지막 프레임 (가장 최근)
            t_last = pred['class_logits'].shape[1] - 1
            cls  = pred['class_logits'][0, t_last].argmax(-1).cpu().numpy()
            doa  = pred['doa_vec'][0, t_last].cpu().numpy()
            conf = torch.sigmoid(pred['confidence'][0, t_last]).cpu().numpy()

            infer_ms = (time.perf_counter() - t0) * 1000
            pred_store.update(cls, doa, conf, infer_ms)

        elapsed = time.perf_counter() - t0
        wait    = max(0.0, interval - elapsed)
        stop_event.wait(timeout=wait)


# =============================================================================
# Real-time visualizer
# =============================================================================

class RealtimeViz:
    """실시간 matplotlib 시각화."""

    def __init__(self, pred_store: PredictionStore, audio_buf: AudioRingBuffer,
                 id_to_label: dict, conf_thresh: float, window_samples: int):
        self.pred_store     = pred_store
        self.audio_buf      = audio_buf
        self.id_to_label    = id_to_label
        self.conf_thresh    = conf_thresh
        self.window_samples = window_samples
        self._start_time    = time.time()

        self._build_figure()

    # ── Figure 구성 ───────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(
            figsize=(FIG_W / 100, FIG_H / 100), dpi=100,
            facecolor='#1a1a2e',
        )
        gs = GridSpec(
            2, 3, figure=self.fig,
            width_ratios=[1.4, 1.4, 0.9], height_ratios=[5, 1],
            hspace=0.08, wspace=0.30,
            left=0.04, right=0.97, top=0.93, bottom=0.08,
        )
        self.ax_polar = self.fig.add_subplot(gs[0, 0], projection='polar',
                                              facecolor='#0d1b2a')
        self.ax_azel  = self.fig.add_subplot(gs[0, 1], facecolor='#0d1b2a')
        self.ax_info  = self.fig.add_subplot(gs[0, 2], facecolor='#0d1b2a')
        self.ax_wave  = self.fig.add_subplot(gs[1, :], facecolor='#0d1b2a')

        self._setup_polar()
        self._setup_azel()
        self.ax_info.axis('off')
        self._setup_wave()

        # 타이틀 텍스트 (동적 업데이트용)
        self._title = self.fig.text(
            0.5, 0.97, 'SLED v3  ·  LIVE  ·  waiting for audio…',
            ha='center', va='top', color='white',
            fontsize=13, fontweight='bold',
        )
        # LIVE 배지
        self.fig.text(
            0.02, 0.97, '⬤ LIVE',
            ha='left', va='top', color='#ef4444',
            fontsize=11, fontweight='bold',
        )

        self._dyn: list = []   # 프레임마다 지워질 동적 artist 목록

    def _setup_polar(self):
        ax = self.ax_polar
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.259, 0.5, 0.707, 0.866, 1.0])
        ax.set_yticklabels(['75°', '60°', '45°', '30°', '0°'],
                            fontsize=6.5, color='#888888')
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(
            ['Front', '45°', 'Right', '135°', 'Back', '225°', 'Left', '315°'],
            color='#cccccc', fontsize=7.5,
        )
        ax.tick_params(colors='#444444')
        ax.spines['polar'].set_color('#334155')
        ax.grid(color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title('Top-Down View  (radius = cos el)',
                      color='#888888', fontsize=8, pad=10)

    def _setup_azel(self):
        ax = self.ax_azel
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(
            ['Front\n0°', 'Right\n90°', 'Back\n180°', 'Left\n270°', '360°'],
            color='#cccccc', fontsize=7,
        )
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_yticklabels(['-90°','-60°','-30°','0°','30°','60°','90°'],
                            color='#888888', fontsize=7)
        ax.axhline(0, color='#556677', linewidth=0.8, linestyle='--')
        ax.grid(color='#334155', linestyle='--', linewidth=0.4, alpha=0.6)
        ax.set_xlabel('Azimuth',   color='#888888', fontsize=7.5)
        ax.set_ylabel('Elevation', color='#888888', fontsize=7.5)
        ax.set_title('Azimuth–Elevation View', color='#888888', fontsize=8)
        for sp in ax.spines.values():
            sp.set_color('#334155')
        ax.tick_params(colors='#444444')

    def _setup_wave(self):
        ax = self.ax_wave
        for sp in ax.spines.values():
            sp.set_color('#334155')
        ax.tick_params(colors='#888888', labelsize=7)
        ax.set_xlabel('Time (s)', color='#888888', fontsize=8)
        self._wave_artists: list = []   # artist handles replaced each frame

    # ── 동적 artist 관리 ──────────────────────────────────────────────────────

    def _clear_dyn(self):
        for a in self._dyn:
            try:
                a.remove()
            except Exception:
                pass
        self._dyn.clear()

    def _add(self, artist):
        self._dyn.append(artist)
        return artist

    # ── 프레임 업데이트 ───────────────────────────────────────────────────────

    def update(self, _frame):
        self._clear_dyn()

        cls, doa, conf, trail, infer_ms, infer_ts, ready = self.pred_store.snapshot()
        elapsed = time.time() - self._start_time

        # 타이틀 갱신
        status = f'{elapsed:.1f} s  |  infer {infer_ms:.0f} ms' if ready \
                 else 'waiting for audio…'
        self._title.set_text(f'SLED v3  ·  LIVE  ·  {status}')

        if ready:
            self._draw_predictions(cls, doa, conf, trail)

        self._draw_waveform()
        self._draw_info(cls, doa, conf, ready)

        return self._dyn

    # ── 예측 그리기 ───────────────────────────────────────────────────────────

    def _draw_predictions(self, cls, doa, conf, trail):
        n_slots = len(cls)
        n_trail = len(trail)

        for s_idx in range(n_slots):
            color = _SLOT_COLORS[s_idx % len(_SLOT_COLORS)]

            # ── 궤적 (오래된 순서로 점점 흐리게) ──────────────────────────────
            for ti, snap in enumerate(trail[:-1]):   # 마지막은 현재점으로 표시
                alpha_frac = (ti + 1) / max(n_trail, 1)
                alpha = 0.1 + 0.4 * alpha_frac
                az_t, el_t = doa_to_az_el(snap['doa'][s_idx])
                r_t = np.cos(np.radians(el_t))
                self._add(self.ax_polar.scatter(
                    np.radians(az_t), r_t,
                    s=50, c=[color], marker='o', alpha=alpha, zorder=4,
                    linewidths=0,
                ))
                self._add(self.ax_azel.scatter(
                    az_t, el_t,
                    s=40, c=[color], marker='o', alpha=alpha, zorder=4,
                    linewidths=0,
                ))

            # ── 현재 예측 ────────────────────────────────────────────────────
            az, el   = doa_to_az_el(doa[s_idx])
            az_rad   = np.radians(az)
            r        = np.cos(np.radians(el))
            c_conf   = float(conf[s_idx])
            cls_id   = int(cls[s_idx])
            label    = self.id_to_label.get(cls_id, f'cls{cls_id}')
            active   = c_conf >= self.conf_thresh
            alpha_pt = 0.95 if active else max(0.2, c_conf)

            # 극좌표
            self._add(self.ax_polar.scatter(
                az_rad, r, s=240, c=[color], marker='*',
                edgecolors='white', linewidths=0.6, zorder=6, alpha=alpha_pt,
            ))
            if active:
                self._add(self.ax_polar.text(
                    az_rad, r - 0.13,
                    f'{label[:12]}\n{c_conf:.2f}',
                    color=color, fontsize=6, ha='center', va='top', alpha=alpha_pt,
                ))

            # az-el
            self._add(self.ax_azel.scatter(
                az, el, s=220, c=[color], marker='*',
                edgecolors='white', linewidths=0.5, zorder=6, alpha=alpha_pt,
            ))
            if active:
                self._add(self.ax_azel.text(
                    az, el - 6,
                    f'{label[:12]}\n{c_conf:.2f}',
                    color=color, fontsize=6, ha='center', va='top', alpha=alpha_pt,
                ))

        # 범례
        legend_elems = [
            mpatches.Patch(facecolor=_SLOT_COLORS[i],
                           label=f'Slot {i+1}', edgecolor='white', linewidth=0.5)
            for i in range(n_slots)
        ]
        self._add(self.ax_polar.legend(
            handles=legend_elems, loc='lower right',
            fontsize=7, facecolor='#1a1a2e', edgecolor='#334155',
            labelcolor='#dddddd', framealpha=0.85,
        ))

    # ── 파형 그리기 ───────────────────────────────────────────────────────────

    def _draw_waveform(self):
        ax = self.ax_wave
        # Remove only the previous waveform artists (avoids ax.cla() full redraw)
        for a in self._wave_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._wave_artists.clear()

        disp_samples = SAMPLE_RATE * 4   # 4초 표시
        wave  = self.audio_buf.read_display(disp_samples)
        wave_L = wave[0]   # L 채널만 표시

        t_axis = np.arange(len(wave_L)) / SAMPLE_RATE
        fill = ax.fill_between(t_axis, wave_L, 0,
                               color='#3b82f6', alpha=0.65, linewidth=0)
        vl   = ax.axvline(t_axis[-1], color='#ef4444', linewidth=1.8, zorder=5)
        ax.set_xlim(t_axis[0], t_axis[-1])
        peak = max(float(np.abs(wave_L).max()), 1e-3)
        ax.set_ylim(-peak * 1.3, peak * 1.3)
        self._wave_artists.extend([fill, vl])

    # ── 정보 패널 ─────────────────────────────────────────────────────────────

    def _draw_info(self, cls, doa, conf, ready: bool):
        ax = self.ax_info
        ax.cla()
        ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        def _txt(y, s, color='#dddddd', bold=False, size=8):
            return ax.text(
                0.04, y, s, transform=ax.transAxes,
                color=color, fontsize=size,
                fontweight='bold' if bold else 'normal',
                va='top', fontfamily='monospace',
            )

        y = 0.97
        _txt(y, 'PREDICTIONS', '#ffffff', bold=True, size=9); y -= 0.07

        if not ready:
            _txt(y, '  (대기 중…)', '#888888'); return

        for s_idx in range(len(cls)):
            color  = _SLOT_COLORS[s_idx % len(_SLOT_COLORS)]
            az, el = doa_to_az_el(doa[s_idx])
            c_conf = float(conf[s_idx])
            cls_id = int(cls[s_idx])
            label  = self.id_to_label.get(cls_id, f'cls{cls_id}')
            active = c_conf >= self.conf_thresh
            tc     = color if active else '#777777'

            _txt(y, f'Slot {s_idx+1}  {label[:16]}', tc);          y -= 0.055
            _txt(y, f'  az={az:6.1f}°  el={el:+5.1f}°  p={c_conf:.2f}', tc)
            y -= 0.065

        y -= 0.02
        _txt(y, 'SETTINGS', '#aaaaaa', bold=True, size=8);        y -= 0.06
        _txt(y, f'  thresh = {self.conf_thresh:.2f}', '#888888'); y -= 0.05
        _txt(y, f'  window = {self.window_samples // SAMPLE_RATE:.2f} s',
              '#888888')

    # ── FuncAnimation 실행 ────────────────────────────────────────────────────

    def run(self):
        self._anim = animation.FuncAnimation(
            self.fig,
            self.update,
            interval   = int(1000 / DISPLAY_FPS),
            blit       = False,
            cache_frame_data = False,
        )
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SLED v3 실시간 binaural 시각화',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--list-devices', action='store_true',
                        help='사용 가능한 오디오 장치 목록을 출력하고 종료')
    parser.add_argument('--ckpt',          default=None,
                        help='모델 체크포인트 경로')
    parser.add_argument('--sofa-path',     default='./hrtf/p0001.sofa')
    parser.add_argument('--class-map',     default=None,
                        help='class_map.json 경로 (레이블 표시용)')
    parser.add_argument('--audio-device',  type=int, default=None,
                        help='sounddevice 입력 장치 번호 (기본: 시스템 기본값)')
    parser.add_argument('--conf-thresh',   type=float, default=0.35)
    parser.add_argument('--window-frames', type=int,   default=48,
                        help='추론 윈도우 프레임 수 (48 × 20ms = 0.96s)')
    parser.add_argument('--d-model',       type=int,   default=256)
    parser.add_argument('--n-classes',     type=int,   default=None,
                        help='클래스 수 (미지정 시 체크포인트에서 자동 추출)')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── 장치 목록 출력 ────────────────────────────────────────────────────────
    if args.list_devices:
        print(sd.query_devices())
        return

    if args.ckpt is None:
        parser.error('--ckpt 가 필요합니다.')

    # ── 체크포인트 로드 ───────────────────────────────────────────────────────
    print(f'[MODEL] 체크포인트 로드: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=args.device)

    # 체크포인트에서 메타데이터 추출
    use_hrtf_corr = ckpt.get('use_hrtf_corr', True)
    use_ild       = ckpt.get('use_ild',        True)
    use_ipd       = ckpt.get('use_ipd',        True)

    # n_classes: 명시 인자 > 체크포인트 > state_dict에서 추론
    if args.n_classes is not None:
        n_classes = args.n_classes
    elif 'n_classes' in ckpt:
        n_classes = ckpt['n_classes']
    else:
        # state_dict에서 heads.class_head.weight 크기로 추론
        state = ckpt['model']
        if 'heads.class_head.weight' in state:
            n_classes = state['heads.class_head.weight'].shape[0]
        else:
            n_classes = 209   # 기본값
    print(f'        n_classes={n_classes}  epoch={ckpt.get("epoch","?")}  '
          f'val_loss={ckpt.get("val_loss", float("nan")):.4f}')

    model = SLEDv3(
        sofa_path     = os.path.abspath(args.sofa_path),
        d_model       = args.d_model,
        n_classes     = n_classes,
        use_hrtf_corr = use_hrtf_corr,
        use_ild       = use_ild,
        use_ipd       = use_ipd,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'        파라미터: {sum(p.numel() for p in model.parameters()):,}')

    # ── 클래스맵 ──────────────────────────────────────────────────────────────
    id_to_label = build_id_to_label(args.class_map)

    # ── 오디오 입력 장치 확인 ─────────────────────────────────────────────────
    dev_info = sd.query_devices(args.audio_device, 'input')
    n_ch_in  = min(int(dev_info['max_input_channels']), 2)
    print(f'[AUDIO] 장치: {dev_info["name"]}  '
          f'(채널 {n_ch_in} → 사용 {n_ch_in})')

    # ── 공유 객체 ─────────────────────────────────────────────────────────────
    window_samples = args.window_frames * HOP_SAMPLES
    audio_buf   = AudioRingBuffer(n_channels=2)
    pred_store  = PredictionStore(n_slots=3)
    stop_event  = threading.Event()
    # 콜백은 여기에만 push; consumer_thread가 ring buffer로 옮김
    chunk_queue: collections.deque = collections.deque(maxlen=200)

    # ── sounddevice 콜백 (최대한 가볍게) ──────────────────────────────────────
    def audio_callback(indata: np.ndarray, frames, t_info, status):
        """PortAudio 내부 thread에서 호출. GIL 대기 최소화."""
        if status:
            # status 출력은 overflow 상황에서 또 block 유발 → 무시
            pass
        chunk_queue.append(indata.copy())   # deque.append는 CPython에서 thread-safe

    # ── consumer thread: chunk_queue → ring buffer ────────────────────────────
    def consumer_worker():
        mono_flag = (n_ch_in == 1)
        while not stop_event.is_set():
            while chunk_queue:
                raw = chunk_queue.popleft()   # [frames, n_ch_in]
                data = raw.T.astype(np.float32)   # [n_ch_in, frames]
                if mono_flag:
                    data = np.repeat(data, 2, axis=0)
                audio_buf.write(data)
            time.sleep(0.005)   # 5ms 대기 후 재확인

    consumer_thread = threading.Thread(
        target=consumer_worker, daemon=True, name='sled-consumer'
    )
    consumer_thread.start()

    # ── 추론 thread 시작 ──────────────────────────────────────────────────────
    infer_thread = threading.Thread(
        target=inference_worker,
        args=(model, audio_buf, pred_store, window_samples,
              args.device, stop_event),
        daemon=True,
        name='sled-infer',
    )
    infer_thread.start()

    # ── 오디오 스트림 열기 ────────────────────────────────────────────────────
    stream = sd.InputStream(
        device      = args.audio_device,
        channels    = n_ch_in,
        samplerate  = SAMPLE_RATE,
        dtype       = 'float32',
        blocksize   = HOP_SAMPLES * 2,   # 40ms — 콜백이 가벼우므로 짧게 유지
        callback    = audio_callback,
        latency     = 'high',
    )

    # ── 시각화 (main thread) ──────────────────────────────────────────────────
    viz = RealtimeViz(
        pred_store     = pred_store,
        audio_buf      = audio_buf,
        id_to_label    = id_to_label,
        conf_thresh    = args.conf_thresh,
        window_samples = window_samples,
    )

    try:
        with stream:
            print('[STREAM] 오디오 스트림 시작. 창을 닫으면 종료됩니다.')
            viz.run()   # plt.show() 블록 → 창 닫으면 반환
    except KeyboardInterrupt:
        print('\n[STOP] Ctrl+C')
    finally:
        stop_event.set()
        infer_thread.join(timeout=2.0)
        print('[DONE]')


if __name__ == '__main__':
    main()
