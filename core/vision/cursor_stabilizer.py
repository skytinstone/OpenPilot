"""
CursorStabilizer — 아시아인 맞춤 EAR 필터 + 1 Euro Filter
============================================================
동양인 사용자의 좁은 눈 특성에 최적화된 커서 안정화 시스템.

구성 요소:
  1. EAR(Eye Aspect Ratio) 깜빡임 필터:
     - 아시아인 맞춤 낮은 임계값 (0.15)
     - 깜빡임 시 커서 Freeze (점프 방지)
     - 연속 프레임 카운터로 오인 최소화

  2. 1 Euro Filter (Casiez et al., 2012):
     - 속도 적응형 스무딩: 느리면 떨림 제거, 빠르면 반응성 유지
     - EMA 대비 장점: 정밀 클릭과 빠른 시선 이동 모두 지원
     - min_cutoff/beta 파라미터로 개인 튜닝 가능

  3. Head Pose 보정:
     - cv2.solvePnP로 머리 회전(Yaw, Pitch, Roll) 추정
     - 머리 기울임에 따른 시선 오프셋 상쇄
"""
import math
import time
import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════
#  1 Euro Filter (Casiez et al., CHI 2012)
# ══════════════════════════════════════════════════════════════════

class _LowPassFilter:
    """지수 스무딩 저역 필터"""

    def __init__(self, alpha: float = 1.0):
        self._y: Optional[float] = None
        self._alpha = alpha

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        a = alpha if alpha is not None else self._alpha
        if self._y is None:
            self._y = value
        else:
            self._y = a * value + (1.0 - a) * self._y
        return self._y

    def reset(self):
        self._y = None

    @property
    def last(self) -> Optional[float]:
        return self._y


class OneEuroFilter:
    """
    1 Euro Filter — 속도 적응형 노이즈 필터

    핵심 원리:
      - 입력 신호의 변화 속도를 측정
      - 느릴 때: cutoff↓ → 강한 스무딩 (떨림 제거)
      - 빠를 때: cutoff↑ → 약한 스무딩 (빠른 반응)

    파라미터 튜닝 가이드:
    ┌─────────────┬────────────────────────────────────────────┐
    │ 파라미터     │ 효과                                       │
    ├─────────────┼────────────────────────────────────────────┤
    │ min_cutoff  │ 정지 시 스무딩 강도. 낮을수록 부드러움.      │
    │             │ 너무 낮으면 반응 느림. 권장: 0.5~2.0        │
    ├─────────────┼────────────────────────────────────────────┤
    │ beta        │ 속도 반응 민감도. 높을수록 빠른 움직임에     │
    │             │ 즉시 반응. 너무 높으면 떨림. 권장: 0.01~0.1 │
    ├─────────────┼────────────────────────────────────────────┤
    │ d_cutoff    │ 속도 추정 스무딩. 보통 1.0 고정.            │
    └─────────────┴────────────────────────────────────────────┘

    아시아인 사용자 권장 기본값:
      min_cutoff=1.2  (안정적, 눈이 작아 노이즈가 더 큼)
      beta=0.05       (적당한 반응성)
    """

    def __init__(self,
                 min_cutoff: float = 1.2,
                 beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()
        self._last_time: Optional[float] = None

    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        t = timestamp if timestamp is not None else time.time()

        if self._last_time is None:
            self._last_time = t
            self._x_filter.filter(value)
            self._dx_filter.filter(0.0)
            return value

        dt = t - self._last_time
        if dt <= 0:
            dt = 1e-6
        self._last_time = t

        # 속도(미분) 추정
        prev = self._x_filter.last or value
        dx = (value - prev) / dt

        # 속도 스무딩
        alpha_d = self._alpha(dt, self._d_cutoff)
        edx = abs(self._dx_filter.filter(dx, alpha_d))

        # 적응형 cutoff: 빠를수록 cutoff 증가
        cutoff = self._min_cutoff + self._beta * edx

        # 값 스무딩
        alpha = self._alpha(dt, cutoff)
        return self._x_filter.filter(value, alpha)

    def reset(self):
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


# ══════════════════════════════════════════════════════════════════
#  EAR (Eye Aspect Ratio) — 깜빡임 감지
# ══════════════════════════════════════════════════════════════════

class BlinkDetector:
    """
    아시아인 맞춤형 EAR 기반 깜빡임 감지기.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
      p1~p6: 눈 윤곽 6개 랜드마크 (MediaPipe 기준)

    아시아인 맞춤 파라미터:
    ┌──────────────────┬────────────────────────────────────────┐
    │ 파라미터          │ 설명                                   │
    ├──────────────────┼────────────────────────────────────────┤
    │ ear_threshold    │ 이 값 이하 = 눈 감김.                  │
    │                  │ 서양인: 0.21, 아시아인: 0.15 (눈이 작음) │
    ├──────────────────┼────────────────────────────────────────┤
    │ consec_frames    │ 연속 N프레임 이하여야 깜빡임 인정.      │
    │                  │ 1프레임만 낮아도 오인 방지: 3프레임 필요  │
    ├──────────────────┼────────────────────────────────────────┤
    │ reopen_threshold │ 눈을 다시 떴다고 판정하는 EAR 값.       │
    │                  │ ear_threshold보다 약간 높게 (히스테리시스)│
    └──────────────────┴────────────────────────────────────────┘
    """

    # MediaPipe 눈 윤곽 6포인트 인덱스 (EAR 계산용)
    # 왼눈: p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    # 오른눈: p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
    LEFT_EAR_IDX  = (362, 385, 387, 263, 373, 380)
    RIGHT_EAR_IDX = (33,  160, 158, 133, 153, 144)

    def __init__(self,
                 ear_threshold: float = 0.15,
                 consec_frames: int = 3,
                 reopen_threshold: float = 0.19):
        """
        Args:
            ear_threshold:    깜빡임 판정 EAR (아시아인: 0.15)
            consec_frames:    연속 프레임 수 (3 = 약 100ms @30fps)
            reopen_threshold: 눈 떰 판정 EAR (히스테리시스)
        """
        self._threshold = ear_threshold
        self._consec = consec_frames
        self._reopen = reopen_threshold
        self._counter = 0           # EAR < threshold 연속 프레임 수
        self._is_blinking = False
        self._ear_history: deque = deque(maxlen=30)  # EAR 히스토리 (자동 임계값용)

    def update(self, landmarks) -> bool:
        """
        매 프레임 호출. landmarks = MediaPipe 478 랜드마크.
        Returns: True = 현재 깜빡임 중 (커서 freeze 필요)
        """
        left_ear  = self._compute_ear(landmarks, self.LEFT_EAR_IDX)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EAR_IDX)
        avg_ear = (left_ear + right_ear) / 2.0

        self._ear_history.append(avg_ear)

        if avg_ear < self._threshold:
            self._counter += 1
            if self._counter >= self._consec:
                self._is_blinking = True
        else:
            if self._is_blinking and avg_ear > self._reopen:
                self._is_blinking = False
            self._counter = 0

        return self._is_blinking

    def _compute_ear(self, landmarks, indices) -> float:
        """
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        p1, p2, p3, p4, p5, p6 = indices
        try:
            def pt(idx):
                lm = landmarks[idx]
                if hasattr(lm, 'x'):
                    return np.array([lm.x, lm.y])
                return np.array([lm[0], lm[1]])

            v1 = np.linalg.norm(pt(p2) - pt(p6))
            v2 = np.linalg.norm(pt(p3) - pt(p5))
            h  = np.linalg.norm(pt(p1) - pt(p4))
            return (v1 + v2) / (2.0 * h + 1e-6)
        except (IndexError, AttributeError):
            return 0.3  # 안전 기본값 (눈 떠있음으로 간주)

    @property
    def is_blinking(self) -> bool:
        return self._is_blinking

    @property
    def current_ear(self) -> float:
        return self._ear_history[-1] if self._ear_history else 0.3

    @property
    def adaptive_threshold(self) -> float:
        """최근 EAR 히스토리 기반 적응형 임계값 (개인차 보정)"""
        if len(self._ear_history) < 15:
            return self._threshold
        # 눈 뜬 상태의 중앙값 × 0.6 → 개인 맞춤 임계값
        sorted_ears = sorted(self._ear_history, reverse=True)
        open_median = sorted_ears[len(sorted_ears) // 4]  # 상위 25% 중앙
        return max(0.10, open_median * 0.6)


# ══════════════════════════════════════════════════════════════════
#  Head Pose Estimator — 3D 머리 자세 추정
# ══════════════════════════════════════════════════════════════════

class HeadPoseEstimator:
    """
    cv2.solvePnP 기반 3D 머리 자세 추정.
    Yaw(좌우), Pitch(상하), Roll(기울임) 을 계산하여
    시선 오프셋에서 머리 움직임 성분을 상쇄.

    사용하는 6개 랜드마크 (표준 3D 모델 기준):
      코끝(1), 턱(152), 왼눈 모서리(263), 오른눈 모서리(33),
      입 왼쪽(287), 입 오른쪽(57)
    """

    # 3D 모델 포인트 (표준 얼굴 비율, 미터 단위 축소)
    _MODEL_3D = np.array([
        (0.0,    0.0,    0.0),     # 코끝
        (0.0,   -0.33,  -0.065),   # 턱
        (-0.225,  0.17,  -0.135),  # 왼눈 외측
        (0.225,   0.17,  -0.135),  # 오른눈 외측
        (-0.15,  -0.15,  -0.125),  # 입 왼쪽
        (0.15,   -0.15,  -0.125),  # 입 오른쪽
    ], dtype=np.float64)

    # 대응 MediaPipe 랜드마크 인덱스
    _LM_IDX = [1, 152, 263, 33, 287, 57]

    def __init__(self):
        self._yaw   = 0.0
        self._pitch = 0.0
        self._roll  = 0.0
        # 카메라 매트릭스 캐시 (프레임 크기별)
        self._cam_matrix_cache: dict = {}

    def update(self, landmarks, frame_w: int, frame_h: int):
        """
        매 프레임 호출. landmarks = MediaPipe 478 랜드마크.
        """
        try:
            pts_2d = []
            for idx in self._LM_IDX:
                lm = landmarks[idx]
                if hasattr(lm, 'x'):
                    pts_2d.append([lm.x * frame_w, lm.y * frame_h])
                else:
                    pts_2d.append([lm[0] * frame_w, lm[1] * frame_h])

            pts_2d = np.array(pts_2d, dtype=np.float64)

            # 카메라 내부 파라미터 (근사값)
            key = (frame_w, frame_h)
            if key not in self._cam_matrix_cache:
                focal = frame_w
                center = (frame_w / 2, frame_h / 2)
                self._cam_matrix_cache[key] = np.array([
                    [focal, 0,     center[0]],
                    [0,     focal, center[1]],
                    [0,     0,     1.0],
                ], dtype=np.float64)

            cam_matrix = self._cam_matrix_cache[key]
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                self._MODEL_3D, pts_2d,
                cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                # 오일러 각도 추출
                sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
                if sy > 1e-6:
                    self._pitch = math.atan2(rmat[2, 1], rmat[2, 2])
                    self._yaw   = math.atan2(-rmat[2, 0], sy)
                    self._roll  = math.atan2(rmat[1, 0], rmat[0, 0])
                else:
                    self._pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
                    self._yaw   = math.atan2(-rmat[2, 0], sy)
                    self._roll  = 0.0

        except Exception:
            pass  # 실패 시 이전 값 유지

    def compensate_gaze(self, gaze_x: float, gaze_y: float,
                        yaw_gain: float = 0.3,
                        pitch_gain: float = 0.25) -> Tuple[float, float]:
        """
        시선 오프셋에서 머리 움직임 성분 차감.

        Args:
            gaze_x, gaze_y: 원본 시선 오프셋
            yaw_gain:       Yaw 보정 강도 (0.2~0.5, 높을수록 보정 강함)
            pitch_gain:     Pitch 보정 강도

        Returns:
            보정된 (gaze_x, gaze_y)
        """
        # 머리 회전량(rad)을 시선 오프셋 스케일로 변환
        # Yaw → X축, Pitch → Y축 상쇄
        comp_x = gaze_x - self._yaw * yaw_gain
        comp_y = gaze_y - self._pitch * pitch_gain
        return comp_x, comp_y

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self._yaw)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self._pitch)

    @property
    def roll_deg(self) -> float:
        return math.degrees(self._roll)


# ══════════════════════════════════════════════════════════════════
#  CursorStabilizer — 통합 커서 안정화
# ══════════════════════════════════════════════════════════════════

class CursorStabilizer:
    """
    모든 안정화 모듈을 통합한 커서 좌표 안정화기.

    파이프라인:
      raw gaze → Head Pose 보정 → Blink Freeze → 1 Euro Filter → screen coords
    """

    def __init__(self,
                 screen_w: int,
                 screen_h: int,
                 # 1 Euro Filter 파라미터
                 min_cutoff: float = 1.2,
                 beta: float = 0.05,
                 # EAR 파라미터 (아시아인 맞춤)
                 ear_threshold: float = 0.15,
                 ear_consec: int = 3,
                 # Head Pose
                 head_pose_enabled: bool = True,
                 yaw_gain: float = 0.3,
                 pitch_gain: float = 0.25):

        self.screen_w = screen_w
        self.screen_h = screen_h

        # 1 Euro Filter (X, Y 각각)
        self._filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        # 깜빡임 감지
        self._blink = BlinkDetector(
            ear_threshold=ear_threshold,
            consec_frames=ear_consec,
        )

        # 머리 자세 보정
        self._head_pose = HeadPoseEstimator() if head_pose_enabled else None
        self._yaw_gain = yaw_gain
        self._pitch_gain = pitch_gain

        # 마지막 유효 좌표 (깜빡임 시 freeze 용)
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

        # 디버그 정보
        self._debug_info: dict = {}

    def stabilize(self, raw_x: float, raw_y: float,
                  landmarks=None,
                  frame_w: int = 0, frame_h: int = 0,
                  timestamp: Optional[float] = None) \
            -> Tuple[int, int]:
        """
        원시 화면 좌표 → 안정화된 화면 좌표.

        Args:
            raw_x, raw_y:  캘리브레이션 적용 후 원시 스크린 좌표
            landmarks:     MediaPipe 478 랜드마크 (깜빡임/헤드포즈용)
            frame_w, frame_h: 카메라 프레임 크기
            timestamp:     현재 시각 (None이면 time.time())

        Returns:
            (screen_x, screen_y) 정수 좌표
        """
        t = timestamp or time.time()
        is_blink = False

        # ── 1. 깜빡임 감지 → Freeze ──────────────────────────
        if landmarks is not None:
            is_blink = self._blink.update(landmarks)

        if is_blink and self._last_x is not None:
            # 깜빡임 중: 마지막 유효 위치 유지
            self._debug_info = {
                "blink": True,
                "ear": self._blink.current_ear,
                "head_yaw": 0.0, "head_pitch": 0.0,
            }
            return int(self._last_x), int(self._last_y)

        # ── 2. Head Pose 보정 (선택) ──────────────────────────
        sx, sy = raw_x, raw_y
        head_yaw, head_pitch = 0.0, 0.0

        if self._head_pose and landmarks is not None and frame_w > 0:
            self._head_pose.update(landmarks, frame_w, frame_h)
            head_yaw = self._head_pose.yaw_deg
            head_pitch = self._head_pose.pitch_deg

            # 시선 좌표에서 머리 회전 성분 차감
            # yaw → X축 이동, pitch → Y축 이동
            sx -= head_yaw * self._yaw_gain * (self.screen_w / 90.0)
            sy -= head_pitch * self._pitch_gain * (self.screen_h / 90.0)

        # ── 3. 1 Euro Filter 적용 ────────────────────────────
        fx = self._filter_x.filter(sx, t)
        fy = self._filter_y.filter(sy, t)

        # 클리핑
        fx = max(0.0, min(self.screen_w - 1, fx))
        fy = max(0.0, min(self.screen_h - 1, fy))

        self._last_x = fx
        self._last_y = fy

        self._debug_info = {
            "blink": False,
            "ear": self._blink.current_ear,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
        }

        return int(fx), int(fy)

    def reset(self):
        """필터 상태 초기화"""
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_x = None
        self._last_y = None

    @property
    def is_blinking(self) -> bool:
        return self._blink.is_blinking

    @property
    def debug_info(self) -> dict:
        return self._debug_info
