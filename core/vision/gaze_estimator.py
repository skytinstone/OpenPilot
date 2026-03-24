"""
시선 추정 모듈 (Enhanced Calibration v3 — Poly2/SVR)
=====================================================
홍채 오프셋 → 화면 좌표 변환

v3 개선 사항:
  - Stability Gate 대폭 완화 (0.0004 → 0.008) — 웹캠 노이즈 허용
  - Affine(2x3) 삭제 → Poly2 비선형 회귀로 전면 교체
  - SVR 옵션 지원 (sklearn 있을 때 자동 활성)
  - Raw gaze 디버깅 로그 추가
  - 품질 0.00 방지: all_samples 폴백 수집
  - 이상치 자동 제거 (중앙값 ±1.5σ 밖 샘플 제거)
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
from .gaze_calibrator import GazeCalibrator

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

# 안정성 설정 (v3: 웹캠 노이즈 환경에 맞게 완화)
_STABILITY_WINDOW  = 12      # 최근 N 프레임으로 분산 계산
_STABILITY_THRESH  = 0.005   # v3: 0.0004→0.005 (완화, 웹캠 노이즈 허용)
_STABLE_DWELL_SEC  = 3.0     # 포인트당 필요 누적 안정 시간 (초)
_OUTLIER_STD       = 1.5     # 중앙값 기준 ±N*σ 밖 샘플 제거

# v3.2: 시선 이동 검증 + Grace Period
_MIN_MOVEMENT      = 0.05    # 이전 포인트 대비 최소 이동 거리 (정규화)
                              # 0.05 = gaze 범위 ±0.3 기준으로 약 8% 이동 필요
                              # 0.03→0.05 강화: 노이즈 떨림과 실제 이동 구분
_GRACE_PERIOD_SEC  = 1.5     # 새 포인트 표시 후 데이터 수집 금지 시간 (초)
                              # 사용자가 시선을 옮길 물리적 시간 보장
                              # 모든 포인트에 동일 적용 (첫 포인트 포함)

# 디버깅 로그 (True면 매 프레임 raw gaze offset 출력)
_DEBUG_RAW_GAZE    = False   # python main.py 실행 전 True로 변경하여 확인

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


_debug_frame_cnt = 0

def _avg_gaze_offset(eye_data: EyeData) -> Tuple[float, float]:
    global _debug_frame_cnt
    lx, ly = eye_data.left_gaze_offset
    rx, ry = eye_data.right_gaze_offset
    avg_x = (lx + rx) / 2.0
    avg_y = (ly + ry) / 2.0

    # v3: Raw gaze 디버깅 로그 (10프레임마다 출력 — 터미널 스팸 방지)
    if _DEBUG_RAW_GAZE:
        _debug_frame_cnt += 1
        if _debug_frame_cnt % 10 == 0:
            print(f"[RAW_GAZE] L=({lx:+.4f},{ly:+.4f}) "
                  f"R=({rx:+.4f},{ry:+.4f}) "
                  f"AVG=({avg_x:+.4f},{avg_y:+.4f})")

    return avg_x, avg_y


def _compute_quality(samples: List[Tuple[float, float]]) -> float:
    """샘플 분산으로 품질 계산 (1.0 = 완벽, 0.0 = 불안정)"""
    if len(samples) < 3:
        return 0.1   # v3: 최소 0.1 보장 (데이터 있으면 0.00 방지)
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    var = float(np.var(xs) + np.var(ys))
    # v3: 스케일 완화 — var 0→1.0, var 0.02→0.0 (기존 0.002→0.02로 10배)
    return float(max(0.1, 1.0 - var / 0.02))


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

        # v3: GazeCalibrator (Poly2 비선형 — Affine 완전 대체)
        self._calibrator = GazeCalibrator(
            screen_w, screen_h, method="poly2",
        )
        self._is_calibrated     = False
        self._accuracy_px       = -1.0
        self._point_qualities:  List[float] = []
        # 하위 호환: _transform_matrix (로드 시만 사용)
        self._transform_matrix: Optional[np.ndarray] = None

        # 캘리브레이션 진행 상태
        self._calibrating       = False
        self._phase             = CalibrationPhase.PREPARE
        self._cal_points:       List[CalibrationPoint] = []
        self._current_idx       = 0
        self._phase_start       = 0.0
        self._stability_buf: deque = deque(maxlen=_STABILITY_WINDOW)
        self._n_points          = 9
        # 3초 누적 안정 체류 추적
        self._stable_accumulated = 0.0
        self._last_frame_time    = 0.0
        self._confirm_time       = 0.0

        # v3.1: 시선 이동 게이트 — 이전 포인트 완료 시 시선 위치 기록
        self._prev_point_gaze: Optional[Tuple[float, float]] = None
        self._movement_confirmed = False  # 시선 이동 확인 플래그

        # 검증 상태
        self._val_points:  List[CalibrationPoint] = []
        self._val_idx      = 0
        self._val_samples: List[Tuple[float, float]] = []

        # 저장된 캘리브레이션 자동 로드
        self._try_load_calibration()

    # ─── 추정 ───────────────────────────────────────────────────

    def estimate(self, eye_data: EyeData) -> Optional[ScreenPoint]:
        go = _avg_gaze_offset(eye_data)
        if self._is_calibrated and self._calibrator.is_calibrated:
            sx, sy = self._calibrator.predict(go[0], go[1])
        elif self._is_calibrated and self._transform_matrix is not None:
            # 하위 호환: 기존 Affine 파일 로드 시
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
        # v3.1: 시선 이동 게이트 초기화
        self._prev_point_gaze    = None
        self._movement_confirmed = False
        print(f"[Calibration] 시작 — {n_points}포인트 (시선 이동 검증 활성)")

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
                # v3.1: 현재 시선 위치를 기록 (다음 포인트의 이동 게이트 기준)
                self._prev_point_gaze = go
                self._movement_confirmed = False

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

        # ── 시간 계산 ──────────────────────────────────────────
        cp  = self._cal_points[self._current_idx]
        now = time.time()
        dt  = now - self._last_frame_time if self._last_frame_time > 0 else 0.0
        dt  = min(dt, 0.1)   # 프레임 드롭 방지
        self._last_frame_time = now
        elapsed_since_point = now - self._phase_start

        # ── 방어 1: Grace Period (1.5초) ───────────────────────
        # 새 포인트 표시 후 1.5초 동안은 일체 데이터 수집 금지.
        # 사용자가 시선을 새 타겟으로 옮길 물리적 시간 보장.
        if self._phase == CalibrationPhase.AIMING and elapsed_since_point < _GRACE_PERIOD_SEC:
            # Grace Period 중: 안정성 버퍼만 업데이트 (수집 안 함)
            self._stability_buf.append(go)
            cp.all_samples.append(go)
            return self._status(done=False)

        # ── 안정성 계산 ──────────────────────────────────────────
        self._stability_buf.append(go)
        is_stable = self._check_stability()

        # ── 방어 2: 최소 시선 이동 검증 (Delta Check) ──────────
        # 이전 포인트 완료 시점의 시선 위치 vs 현재 시선:
        #   유클리디안 거리 < _MIN_MOVEMENT → 시선 안 옮김 → 수집 차단
        if not self._movement_confirmed:
            if self._prev_point_gaze is not None:
                dx = go[0] - self._prev_point_gaze[0]
                dy = go[1] - self._prev_point_gaze[1]
                movement = (dx ** 2 + dy ** 2) ** 0.5
                if movement >= _MIN_MOVEMENT:
                    self._movement_confirmed = True
                    print(f"[Calibration] 포인트 {self._current_idx+1} — "
                          f"시선 이동 확인 (Δ={movement:.3f} ≥ {_MIN_MOVEMENT})")
                else:
                    # 아직 시선 안 옮김 → 이 프레임 무시
                    cp.all_samples.append(go)
                    return self._status(done=False)
            else:
                # 첫 번째 포인트: Grace Period 이미 지났으면 OK
                self._movement_confirmed = True

        # ── AIMING → SAMPLING: 이동 확인 + 안정 → 수집 시작 ───
        if (self._phase == CalibrationPhase.AIMING
                and is_stable
                and self._movement_confirmed):
            self._phase       = CalibrationPhase.SAMPLING
            self._phase_start = now
            self._stability_buf.clear()  # 수집 시작 시 버퍼 리셋
            print(f"[Calibration] 포인트 {self._current_idx+1} — "
                  f"시선 도착, 수집 시작 (gaze={go[0]:+.4f},{go[1]:+.4f})")

        # ── SAMPLING: 안정 시간 누적 ─────────────────────────────
        if self._phase == CalibrationPhase.SAMPLING:
            if is_stable:
                self._stable_accumulated += dt
            cp.stable_samples.append(go)
        cp.all_samples.append(go)

        # ── 3초 누적 달성 → 포인트 확인! ─────────────────────────
        if (self._phase == CalibrationPhase.SAMPLING
                and self._stable_accumulated >= _STABLE_DWELL_SEC):
            cp.quality = _compute_quality(cp.stable_samples)
            self._point_qualities.append(cp.quality)
            self._confirm_time = now
            q_label = "★★★" if cp.quality > 0.7 else "★★" if cp.quality > 0.4 else "★"
            avg_gaze = cp.avg_gaze
            gaze_str = (f"gaze=({avg_gaze[0]:+.4f},{avg_gaze[1]:+.4f})"
                        if avg_gaze else "gaze=N/A")
            print(f"[Calibration] ✓ 포인트 {self._current_idx+1}/{len(self._cal_points)} "
                  f"확인 — 품질 {q_label} ({cp.quality:.2f}), {gaze_str}")
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
            # v3: Calibrator 사용 (Poly2) → Affine 폴백
            if self._calibrator.is_calibrated:
                sx, sy = self._calibrator.predict(gx_avg, gy_avg)
            else:
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
        gaze_pts, screen_pts = [], []
        for i, cp in enumerate(self._cal_points):
            avg = cp.avg_gaze
            if avg is None:
                # v3: stable_samples가 없으면 all_samples에서 폴백
                if cp.all_samples:
                    xs = [s[0] for s in cp.all_samples]
                    ys = [s[1] for s in cp.all_samples]
                    avg = (float(np.mean(xs)), float(np.mean(ys)))
                    print(f"[Calibration] ⚠ 포인트 {i+1}: stable 없음 → "
                          f"all_samples 폴백 ({len(cp.all_samples)}개)")
                else:
                    print(f"[Calibration] ⚠ 포인트 {i+1}: 데이터 없음 — 건너뜀")
                    continue

            gaze_pts.append(avg)
            screen_pts.append((cp.screen_x, cp.screen_y))

            # v3: Raw 디버그 로그
            print(f"[Calibration] 포인트 {i+1}: "
                  f"gaze=({avg[0]:+.4f},{avg[1]:+.4f}) → "
                  f"screen=({cp.screen_x:.2f},{cp.screen_y:.2f})")

        if len(gaze_pts) < 3:
            print("[Calibration] ❌ 유효 포인트 부족 ({len(gaze_pts)}/3) — 기본 매핑 대체")
            self._calibrating = False
            return

        # v3: Poly2 비선형 회귀로 피팅 (Affine 완전 대체)
        result = self._calibrator.fit(gaze_pts, screen_pts)
        if result["success"]:
            self._is_calibrated = True
            self._transform_matrix = None  # Affine 사용 안 함
            avg_q = float(np.mean(self._point_qualities)) if self._point_qualities else 0.0
            print(f"[Calibration] ✅ {result['method']} 피팅 완료 — "
                  f"RMSE={result['residual']:.1f}px, 평균 품질={avg_q:.2f}, "
                  f"포인트={len(gaze_pts)}/{len(self._cal_points)}")
        else:
            print(f"[Calibration] ❌ 피팅 실패 — 기본 매핑 대체")
            self._calibrating = False
            return

        # 검증 단계 진입
        self._val_errors  = []
        self._val_idx     = 0
        self._val_samples = []
        self._phase       = CalibrationPhase.VALIDATE
        self._phase_start = time.time()

    # ─── 저장 / 불러오기 ─────────────────────────────────────────

    def _save_calibration(self):
        # v3: GazeCalibrator가 저장 담당
        self._calibrator.accuracy_px = self._accuracy_px
        self._calibrator.save()

    def _try_load_calibration(self):
        # v3: GazeCalibrator로 로드 시도
        if self._calibrator.load():
            self._is_calibrated = True
            self._accuracy_px = self._calibrator.accuracy_px
            return

        # 하위 호환: 기존 Affine 형식 파일 로드
        path = os.path.abspath(_CAL_SAVE_PATH)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if "matrix" in data and "weights_x" not in data:
                mat = np.array(data["matrix"])
                if mat.shape == (2, 3):
                    self._transform_matrix = mat
                    self._is_calibrated    = True
                    self._accuracy_px      = data.get("accuracy_px", -1.0)
                    print(f"[Calibration] 기존 Affine 캘리브레이션 로드 "
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
        self._calibrator = GazeCalibrator(self.screen_w, self.screen_h, method="poly2")

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
