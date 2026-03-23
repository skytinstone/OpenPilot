"""
시선 추정 모듈
홍채 오프셋(눈 회전) → 화면 좌표 변환 (EMA 스무딩 + 캘리브레이션)

변환 방식:
  1. 캘리브레이션 없이: 시선 오프셋(gaze_offset) 을 화면 중앙 기준으로 확대 매핑
     - gaze_offset = 홍채가 눈 중심에서 벗어난 정도 (-1 ~ +1)
     - 얼굴을 움직여도 커서가 따라가지 않음; 눈만 움직여야 커서 이동
  2. 캘리브레이션 후: 5포인트 보정으로 정확도 향상
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from .eye_tracker import EyeData


@dataclass
class ScreenPoint:
    x: int
    y: int


@dataclass
class CalibrationPoint:
    screen_x: float       # 화면 목표 좌표 (정규화 0~1)
    screen_y: float
    gaze_samples: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def avg_gaze(self) -> Optional[Tuple[float, float]]:
        if not self.gaze_samples:
            return None
        return (
            sum(s[0] for s in self.gaze_samples) / len(self.gaze_samples),
            sum(s[1] for s in self.gaze_samples) / len(self.gaze_samples),
        )


# 5포인트 캘리브레이션 위치 (정규화 0~1) — 화면 가장자리까지 커버
CALIBRATION_POSITIONS = [
    (0.05, 0.05),  # 좌상단 (극단)
    (0.95, 0.05),  # 우상단 (극단)
    (0.5,  0.5 ),  # 중앙
    (0.05, 0.95),  # 좌하단 (극단)
    (0.95, 0.95),  # 우하단 (극단)
]


def _avg_gaze_offset(eye_data: EyeData) -> Tuple[float, float]:
    """양쪽 눈 시선 오프셋 평균 반환"""
    lx, ly = eye_data.left_gaze_offset
    rx, ry = eye_data.right_gaze_offset
    return (lx + rx) / 2.0, (ly + ry) / 2.0


class GazeEstimator:
    def __init__(self, screen_w: int, screen_h: int, config: dict):
        self.screen_w = screen_w
        self.screen_h = screen_h

        cfg = config.get("eye_tracking", {})
        self._alpha      = cfg.get("smoothing_alpha", 0.2)      # EMA 계수 (낮을수록 부드러움)
        self._dead_zone  = cfg.get("dead_zone_px", 4)           # 미세 떨림 무시 픽셀
        # gaze_offset → 화면 스케일 (클수록 민감, 작을수록 둔감)
        self._gaze_scale_x = cfg.get("gaze_scale_x", 1.6)
        self._gaze_scale_y = cfg.get("gaze_scale_y", 1.6)

        # EMA 상태
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # 캘리브레이션
        self._is_calibrated = False
        self._calibration_points: List[CalibrationPoint] = []
        self._transform_matrix: Optional[np.ndarray] = None

        # 캘리브레이션 중 상태
        self._calibrating = False
        self._current_cal_idx = 0
        self._cal_start_time: Optional[float] = None
        self._cal_dwell_ms = cfg.get("calibration_dwell_ms", 1500)

    # ─── 좌표 변환 ──────────────────────────────────────────────

    def estimate(self, eye_data: EyeData) -> Optional[ScreenPoint]:
        """시선 오프셋 → 스무딩된 화면 좌표 반환"""
        go = _avg_gaze_offset(eye_data)

        if self._is_calibrated and self._transform_matrix is not None:
            sx, sy = self._apply_calibration(go)
        else:
            sx, sy = self._gaze_to_screen(go)

        # EMA 스무딩
        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = sx, sy
        else:
            new_x = self._alpha * sx + (1 - self._alpha) * self._smooth_x
            new_y = self._alpha * sy + (1 - self._alpha) * self._smooth_y

            # Dead zone: 미세 떨림 무시
            if abs(new_x - self._smooth_x) > self._dead_zone:
                self._smooth_x = new_x
            if abs(new_y - self._smooth_y) > self._dead_zone:
                self._smooth_y = new_y

        x = int(np.clip(self._smooth_x, 0, self.screen_w - 1))
        y = int(np.clip(self._smooth_y, 0, self.screen_h - 1))
        return ScreenPoint(x=x, y=y)

    def _gaze_to_screen(self, go: Tuple[float, float]) -> Tuple[float, float]:
        """
        시선 오프셋 → 화면 좌표 (캘리브레이션 없는 기본 매핑)

        gaze_offset 은 눈 안에서 홍채가 얼마나 치우쳤는지를 나타냄:
          0    → 정면 응시 → 화면 중앙
          +0.5 → 오른쪽 응시 → 화면 오른쪽
          -0.5 → 왼쪽 응시 → 화면 왼쪽
        """
        sx = self.screen_w  * (0.5 + go[0] * self._gaze_scale_x)
        sy = self.screen_h  * (0.5 + go[1] * self._gaze_scale_y)
        return sx, sy

    def _apply_calibration(self, go: Tuple[float, float]) -> Tuple[float, float]:
        """캘리브레이션 변환 행렬 적용"""
        src = np.array([[go[0], go[1], 1.0]])
        dst = src @ self._transform_matrix.T
        return dst[0, 0], dst[0, 1]

    # ─── 캘리브레이션 ──────────────────────────────────────────

    def start_calibration(self):
        """캘리브레이션 시작"""
        self._calibrating = True
        self._current_cal_idx = 0
        self._cal_start_time = time.time()
        self._calibration_points = [
            CalibrationPoint(screen_x=p[0], screen_y=p[1])
            for p in CALIBRATION_POSITIONS
        ]
        self._is_calibrated = False
        self._smooth_x = None
        self._smooth_y = None
        print("[Calibration] 시작 — 화면에 표시되는 점을 바라보세요")

    def update_calibration(self, eye_data: EyeData) -> dict:
        """
        캘리브레이션 진행 업데이트
        반환: {"done": bool, "current_idx": int, "total": int, "progress": float}
        """
        if not self._calibrating:
            return {"done": True, "current_idx": 0, "total": len(CALIBRATION_POSITIONS), "progress": 1.0}

        elapsed = (time.time() - self._cal_start_time) * 1000  # ms
        progress = min(elapsed / self._cal_dwell_ms, 1.0)

        # 현재 포인트에 시선 오프셋 샘플 수집
        go = _avg_gaze_offset(eye_data)
        self._calibration_points[self._current_cal_idx].gaze_samples.append(go)

        if elapsed >= self._cal_dwell_ms:
            print(f"[Calibration] 포인트 {self._current_cal_idx + 1}/{len(CALIBRATION_POSITIONS)} 완료")
            self._current_cal_idx += 1
            self._cal_start_time = time.time()

            if self._current_cal_idx >= len(CALIBRATION_POSITIONS):
                self._finish_calibration()
                return {"done": True, "current_idx": self._current_cal_idx,
                        "total": len(CALIBRATION_POSITIONS), "progress": 1.0}

        return {
            "done": False,
            "current_idx": self._current_cal_idx,
            "total": len(CALIBRATION_POSITIONS),
            "progress": progress,
        }

    def _finish_calibration(self):
        """캘리브레이션 완료 — 변환 행렬 계산"""
        src_pts = []  # 시선 오프셋 좌표
        dst_pts = []  # 화면 좌표

        for cp in self._calibration_points:
            avg = cp.avg_gaze
            if avg is None:
                continue
            src_pts.append([avg[0], avg[1], 1.0])
            dst_pts.append([cp.screen_x * self.screen_w, cp.screen_y * self.screen_h])

        if len(src_pts) < 3:
            print("[Calibration] 샘플 부족 — 기본 매핑으로 대체")
            self._calibrating = False
            return

        # 최소제곱법으로 변환 행렬 계산
        src = np.array(src_pts)
        dst = np.array(dst_pts)
        self._transform_matrix, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
        self._transform_matrix = self._transform_matrix.T

        self._is_calibrated = True
        self._calibrating = False
        print("[Calibration] 완료 — 보정 변환 행렬 계산됨")

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def current_calibration_point(self) -> Optional[CalibrationPoint]:
        if self._calibrating and self._current_cal_idx < len(self._calibration_points):
            return self._calibration_points[self._current_cal_idx]
        return None

    def reset(self):
        """스무딩 상태 초기화"""
        self._smooth_x = None
        self._smooth_y = None
