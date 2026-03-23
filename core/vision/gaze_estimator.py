"""
시선 추정 모듈 (Enhanced Calibration v2)
=========================================
홍채 오프셋 → 화면 좌표 변환 (EMA 스무딩 + 고도화 캘리브레이션)

캘리브레이션 개선 사항:
  - 안정성 감지(Stability Gate): 시선이 안정됐을 때만 샘플 수집
  - 이상치 자동 제거 (중앙값 ±1.5σ 밖 샘플 제거)
  - 5/9 포인트 선택
  - 검증 단계 (완료 후 정확도 측정)
  - 캘리브레이션 저장/불러오기
"""
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

from .eye_tracker import EyeData

# ── 캘리브레이션 포인트 그리드 ────────────────────────────────────

# 5포인트: 네 모서리 + 중앙
CALIBRATION_5PT = [
    (0.05, 0.05), (0.95, 0.05),
    (0.50, 0.50),
    (0.05, 0.95), (0.95, 0.95),
]

# 9포인트: 3×3 그리드 (권장 — 더 정확)
CALIBRATION_9PT = [
    (0.05, 0.05), (0.50, 0.05), (0.95, 0.05),
    (0.05, 0.50), (0.50, 0.50), (0.95, 0.50),
    (0.05, 0.95), (0.50, 0.95), (0.95, 0.95),
]

# 기본 (하위 호환)
CALIBRATION_POSITIONS = CALIBRATION_5PT

# 검증 포인트 (캘리브레이션 포인트 사이 위치)
VALIDATION_POSITIONS = [
    (0.25, 0.25), (0.75, 0.25),
    (0.25, 0.75), (0.75, 0.75),
]

# 안정성 설정
_STABILITY_WINDOW  = 12      # 최근 N 프레임으로 분산 계산
_STABILITY_THRESH  = 0.0004  # 이 분산 미만이면 '안정' 판정
_STABLE_DWELL_SEC  = 3.0     # 포인트당 필요 누적 안정 시간 (초)
_OUTLIER_STD       = 1.5     # 중앙값 기준 ±N*σ 밖 샘플 제거

# 저장 경로
_CAL_SAVE_PATH = os.path.join(os.path.dirname(__file__),
                               "../../config/calibration.json")


@dataclass
class ScreenPoint:
    x: int
    y: int


@dataclass
class CalibrationPoint:
    screen_x: float        # 화면 목표 좌표 (정규화 0~1)
    screen_y: float
    all_samples:    List[Tuple[float, float]] = field(default_factory=list)
    stable_samples: List[Tuple[float, float]] = field(default_factory=list)
    quality: float = 0.0   # 0~1 (1 = 완벽)

    @property
    def avg_gaze(self) -> Optional[Tuple[float, float]]:
        """이상치 제거 후 평균"""
        samples = self.stable_samples or self.all_samples
        if not samples:
            return None
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        # 이상치 제거
        xs_f, ys_f = _reject_outliers(xs, ys)
        if not xs_f:
            return float(np.mean(xs)), float(np.mean(ys))
        return float(np.mean(xs_f)), float(np.mean(ys_f))

    @property
    def stable_count(self) -> int:
        return len(self.stable_samples)


def _reject_outliers(xs, ys, n_std=_OUTLIER_STD):
    """중앙값 기준 ±n_std*σ 밖 샘플 제거"""
    if len(xs) < 4:
        return xs, ys
    mx, my = np.median(xs), np.median(ys)
    sx, sy = np.std(xs) + 1e-9, np.std(ys) + 1e-9
    filtered = [(x, y) for x, y in zip(xs, ys)
                if abs(x - mx) < n_std * sx and abs(y - my) < n_std * sy]
    if not filtered:
        return xs, ys
    return [f[0] for f in filtered], [f[1] for f in filtered]


def _avg_gaze_offset(eye_data: EyeData) -> Tuple[float, float]:
    lx, ly = eye_data.left_gaze_offset
    rx, ry = eye_data.right_gaze_offset
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def _compute_quality(samples: List[Tuple[float, float]]) -> float:
    """샘플 분산으로 품질 계산 (1.0 = 완벽, 0.0 = 불안정)"""
    if len(samples) < 3:
        return 0.0
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    var = float(np.var(xs) + np.var(ys))
    # var 0 → quality 1.0, var 0.002 → quality 0.0
    return float(max(0.0, 1.0 - var / 0.002))


# ── CalibrationPhase ─────────────────────────────────────────────

class CalibrationPhase:
    PREPARE  = "prepare"   # 준비 (안내 메시지)
    AIMING   = "aiming"    # 포인트 응시 대기 (안정화)
    SAMPLING = "sampling"  # 안정 샘플 수집 중
    DONE_PT  = "done_pt"   # 한 포인트 완료 (잠깐 표시)
    VALIDATE = "validate"  # 검증 포인트 측정
    RESULT   = "result"    # 최종 결과 표시


# ── GazeEstimator ────────────────────────────────────────────────

class GazeEstimator:
    def __init__(self, screen_w: int, screen_h: int, config: dict):
        self.screen_w = screen_w
        self.screen_h = screen_h

        cfg = config.get("eye_tracking", {})
        self._alpha         = cfg.get("smoothing_alpha", 0.2)
        self._dead_zone     = cfg.get("dead_zone_px", 4)
        self._gaze_scale_x  = cfg.get("gaze_scale_x",  2.5)
        self._gaze_scale_y  = cfg.get("gaze_scale_y", -2.5)  # 음수 = Y반전 보정

        # EMA 상태
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # 캘리브레이션 결과
        self._is_calibrated     = False
        self._transform_matrix: Optional[np.ndarray] = None
        self._accuracy_px       = -1.0   # 검증 단계에서 측정한 오차 (px)
        self._point_qualities:  List[float] = []

        # 캘리브레이션 진행 상태
        self._calibrating       = False
        self._phase             = CalibrationPhase.PREPARE
        self._cal_points:       List[CalibrationPoint] = []
        self._current_idx       = 0
        self._phase_start       = 0.0
        self._stability_buf: deque = deque(maxlen=_STABILITY_WINDOW)
        self._n_points          = 9
        # 3초 누적 안정 체류 추적
        self._stable_accumulated = 0.0   # 누적 안정 시간 (초)
        self._last_frame_time    = 0.0   # 이전 프레임 시각
        self._confirm_time       = 0.0   # 포인트 확인 시각 (이펙트용)

        # 검증 상태
        self._val_points:  List[CalibrationPoint] = []
        self._val_idx      = 0
        self._val_samples: List[Tuple[float, float]] = []

        # 저장된 캘리브레이션 자동 로드
        self._try_load_calibration()

    # ─── 추정 ───────────────────────────────────────────────────

    def estimate(self, eye_data: EyeData) -> Optional[ScreenPoint]:
        go = _avg_gaze_offset(eye_data)
        if self._is_calibrated and self._transform_matrix is not None:
            sx, sy = self._apply_calibration(go)
        else:
            sx, sy = self._gaze_to_screen(go)

        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = sx, sy
        else:
            nx = self._alpha * sx + (1 - self._alpha) * self._smooth_x
            ny = self._alpha * sy + (1 - self._alpha) * self._smooth_y
            if abs(nx - self._smooth_x) > self._dead_zone:
                self._smooth_x = nx
            if abs(ny - self._smooth_y) > self._dead_zone:
                self._smooth_y = ny

        x = int(np.clip(self._smooth_x, 0, self.screen_w - 1))
        y = int(np.clip(self._smooth_y, 0, self.screen_h - 1))
        return ScreenPoint(x=x, y=y)

    def _gaze_to_screen(self, go):
        sx = self.screen_w * (0.5 + go[0] * self._gaze_scale_x)
        sy = self.screen_h * (0.5 + go[1] * self._gaze_scale_y)
        return sx, sy

    def _apply_calibration(self, go):
        src = np.array([[go[0], go[1], 1.0]])
        dst = src @ self._transform_matrix.T
        return dst[0, 0], dst[0, 1]

    # ─── 캘리브레이션 시작 ───────────────────────────────────────

    def start_calibration(self, n_points: int = 9):
        """캘리브레이션 시작. n_points: 5 또는 9"""
        self._n_points = n_points
        positions = CALIBRATION_9PT if n_points == 9 else CALIBRATION_5PT
        self._cal_points = [CalibrationPoint(screen_x=p[0], screen_y=p[1])
                            for p in positions]
        self._current_idx   = 0
        self._is_calibrated = False
        self._calibrating   = True
        self._phase         = CalibrationPhase.PREPARE
        self._phase_start   = time.time()
        self._stability_buf.clear()
        self._point_qualities = []
        self._val_points = []
        self._val_idx    = 0
        self._smooth_x   = None
        self._smooth_y   = None
        print(f"[Calibration] 시작 — {n_points}포인트")

    # ─── 캘리브레이션 업데이트 (매 프레임) ──────────────────────

    def update_calibration(self, eye_data: EyeData) -> dict:
        """
        매 프레임 호출. 반환값:
          {
            "phase": str,
            "done": bool,          # 전체 완료
            "current_idx": int,
            "total": int,
            "stable_count": int,   # 현재 포인트 안정 샘플 수
            "stable_required": int,
            "is_stable": bool,
            "progress": float,     # 0~1 (현재 포인트 진행률)
            "quality": float,      # 현재 포인트 품질 0~1
            "accuracy_px": float,  # 검증 완료 후 설정
            "point_qualities": list,
          }
        """
        if not self._calibrating:
            return self._status(done=True)

        go = _avg_gaze_offset(eye_data)

        # ── 준비 단계 (2초 안내) ─────────────────────────────────
        if self._phase == CalibrationPhase.PREPARE:
            if time.time() - self._phase_start >= 2.0:
                self._phase       = CalibrationPhase.AIMING
                self._phase_start = time.time()
            return self._status(done=False)

        # ── 검증 단계 ────────────────────────────────────────────
        if self._phase == CalibrationPhase.VALIDATE:
            return self._update_validation(go)

        # ── 결과 단계 ────────────────────────────────────────────
        if self._phase == CalibrationPhase.RESULT:
            if time.time() - self._phase_start >= 3.0:
                self._calibrating = False
                self._save_calibration()
                return self._status(done=True)
            return self._status(done=False)

        # ── 포인트 확인 이펙트 표시 (1.5초) ─────────────────────
        if self._phase == CalibrationPhase.DONE_PT:
            if time.time() - self._phase_start >= 1.5:
                self._current_idx += 1
                if self._current_idx >= len(self._cal_points):
                    self._finish_calibration()
                    return self._status(done=False)
                self._phase              = CalibrationPhase.AIMING
                self._phase_start        = time.time()
                self._stable_accumulated = 0.0
                self._last_frame_time    = 0.0
                self._stability_buf.clear()
            return self._status(done=False)

        # ── 안정성 계산 ──────────────────────────────────────────
        self._stability_buf.append(go)
        is_stable = self._check_stability()

        cp  = self._cal_points[self._current_idx]
        now = time.time()
        dt  = now - self._last_frame_time if self._last_frame_time > 0 else 0.0
        dt  = min(dt, 0.1)   # 프레임 드롭 방지
        self._last_frame_time = now

        # AIMING → SAMPLING: 안정되면 진입
        if self._phase == CalibrationPhase.AIMING and is_stable:
            self._phase       = CalibrationPhase.SAMPLING
            self._phase_start = time.time()

        # SAMPLING: 안정 시간 누적 (불안정해도 멈추기만 하고 리셋 안함)
        if self._phase == CalibrationPhase.SAMPLING:
            if is_stable:
                self._stable_accumulated += dt
            cp.stable_samples.append(go)
        cp.all_samples.append(go)

        # 3초 누적 달성 → 포인트 확인!
        if (self._phase == CalibrationPhase.SAMPLING
                and self._stable_accumulated >= _STABLE_DWELL_SEC):
            cp.quality = _compute_quality(cp.stable_samples)
            self._point_qualities.append(cp.quality)
            self._confirm_time = now
            q_label = "★★★" if cp.quality > 0.7 else "★★" if cp.quality > 0.4 else "★"
            print(f"[Calibration] ✓ 포인트 {self._current_idx+1}/{len(self._cal_points)} "
                  f"확인 — 품질 {q_label} ({cp.quality:.2f})")
            self._phase       = CalibrationPhase.DONE_PT
            self._phase_start = now

        return self._status(done=False)

    def _check_stability(self) -> bool:
        if len(self._stability_buf) < _STABILITY_WINDOW:
            return False
        xs = [s[0] for s in self._stability_buf]
        ys = [s[1] for s in self._stability_buf]
        return (float(np.var(xs)) + float(np.var(ys))) < _STABILITY_THRESH

    def _status(self, done: bool) -> dict:
        cp = (self._cal_points[self._current_idx]
              if self._calibrating and self._current_idx < len(self._cal_points)
              else None)
        is_stable = (self._phase == CalibrationPhase.SAMPLING or
                     self._phase == CalibrationPhase.DONE_PT)
        progress  = min(self._stable_accumulated / _STABLE_DWELL_SEC, 1.0)
        quality   = cp.quality if (cp and cp.quality > 0) else \
                    (_compute_quality(cp.stable_samples) if cp and cp.stable_samples else 0.0)

        # validation phase인 경우 validation idx 사용
        cur_idx = (self._val_idx + len(self._cal_points)
                   if self._phase == CalibrationPhase.VALIDATE
                   else self._current_idx)

        return {
            "phase":           self._phase,
            "done":            done,
            "current_idx":     cur_idx,
            "total":           len(self._cal_points),
            "stable_seconds":  self._stable_accumulated,
            "stable_required": _STABLE_DWELL_SEC,
            "is_stable":       is_stable,
            "progress":        progress,
            "quality":         quality,
            "accuracy_px":     self._accuracy_px,
            "point_qualities": list(self._point_qualities),
            "confirm_time":    self._confirm_time,
        }

    # ─── 검증 ────────────────────────────────────────────────────

    def _update_validation(self, go) -> dict:
        if self._val_idx >= len(VALIDATION_POSITIONS):
            self._compute_accuracy()
            self._phase       = CalibrationPhase.RESULT
            self._phase_start = time.time()
            return self._status(done=False)

        self._val_samples.append(go)
        if len(self._val_samples) >= 20:
            xs = [s[0] for s in self._val_samples]
            ys = [s[1] for s in self._val_samples]
            gx_avg = float(np.mean(xs))
            gy_avg = float(np.mean(ys))
            sx, sy = self._apply_calibration((gx_avg, gy_avg))
            vp = VALIDATION_POSITIONS[self._val_idx]
            err = ((sx - vp[0] * self.screen_w) ** 2 +
                   (sy - vp[1] * self.screen_h) ** 2) ** 0.5
            vcp = CalibrationPoint(screen_x=vp[0], screen_y=vp[1])
            vcp.quality = _compute_quality(self._val_samples)
            if not hasattr(self, '_val_errors'):
                self._val_errors = []
            self._val_errors.append(err)
            self._val_points.append(vcp)
            print(f"[Validation] 포인트 {self._val_idx+1} — 오차 {err:.0f}px")
            self._val_idx    += 1
            self._val_samples = []
            self._phase_start = time.time()

        # 각 검증 포인트 3초 대기
        if time.time() - self._phase_start > 3.0 and len(self._val_samples) == 0:
            self._val_idx    += 1
            self._val_samples = []
            self._phase_start = time.time()

        return self._status(done=False)

    def _compute_accuracy(self):
        if hasattr(self, '_val_errors') and self._val_errors:
            self._accuracy_px = float(np.mean(self._val_errors))
            print(f"[Calibration] 평균 오차: {self._accuracy_px:.0f}px")

    # ─── 캘리브레이션 완료 → 행렬 계산 ──────────────────────────

    def _finish_calibration(self):
        src_pts, dst_pts = [], []
        for cp in self._cal_points:
            avg = cp.avg_gaze
            if avg is None:
                continue
            src_pts.append([avg[0], avg[1], 1.0])
            dst_pts.append([cp.screen_x * self.screen_w,
                            cp.screen_y * self.screen_h])

        if len(src_pts) < 3:
            print("[Calibration] 샘플 부족 — 기본 매핑으로 대체")
            self._calibrating = False
            return

        src = np.array(src_pts)
        dst = np.array(dst_pts)
        mat, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
        self._transform_matrix = mat.T
        self._is_calibrated    = True

        avg_q = float(np.mean(self._point_qualities)) if self._point_qualities else 0.0
        print(f"[Calibration] 변환 행렬 계산 완료 — 평균 품질 {avg_q:.2f}")

        # 검증 단계 진입
        self._val_errors  = []
        self._val_idx     = 0
        self._val_samples = []
        self._phase       = CalibrationPhase.VALIDATE
        self._phase_start = time.time()

    # ─── 저장 / 불러오기 ─────────────────────────────────────────

    def _save_calibration(self):
        if self._transform_matrix is None:
            return
        data = {
            "matrix":         self._transform_matrix.tolist(),
            "accuracy_px":    round(self._accuracy_px, 1),
            "n_points":       self._n_points,
            "point_qualities": [round(q, 3) for q in self._point_qualities],
            "created_at":     time.strftime("%Y-%m-%d %H:%M"),
        }
        path = os.path.abspath(_CAL_SAVE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Calibration] 저장됨 → {path}")

    def _try_load_calibration(self):
        path = os.path.abspath(_CAL_SAVE_PATH)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            mat = np.array(data["matrix"])
            if mat.shape == (2, 3):
                self._transform_matrix = mat
                self._is_calibrated    = True
                self._accuracy_px      = data.get("accuracy_px", -1.0)
                print(f"[Calibration] 저장된 캘리브레이션 로드 "
                      f"(정확도 {self._accuracy_px:.0f}px, "
                      f"{data.get('created_at','')})")
        except Exception as e:
            print(f"[Calibration] 로드 실패: {e}")

    def delete_saved_calibration(self):
        path = os.path.abspath(_CAL_SAVE_PATH)
        if os.path.exists(path):
            os.remove(path)
            print("[Calibration] 저장된 캘리브레이션 삭제됨")
        self._is_calibrated    = False
        self._transform_matrix = None

    # ─── 프로퍼티 ────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def calibration_phase(self) -> str:
        return self._phase

    @property
    def accuracy_px(self) -> float:
        return self._accuracy_px

    @property
    def current_calibration_point(self) -> Optional[CalibrationPoint]:
        if self._calibrating and self._current_idx < len(self._cal_points):
            return self._cal_points[self._current_idx]
        return None

    @property
    def current_validation_point(self) -> Optional[Tuple[float, float]]:
        if self._phase == CalibrationPhase.VALIDATE and self._val_idx < len(VALIDATION_POSITIONS):
            return VALIDATION_POSITIONS[self._val_idx]
        return None

    def reset(self):
        self._smooth_x = None
        self._smooth_y = None
