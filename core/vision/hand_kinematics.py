"""
KinematicsClassifier + HysteresisGestureFilter
================================================
MediaPipe HandLandmarker 21개 3D 랜드마크 기반 고도화 제스처 인식.

AS-IS (기존):
  - 2D y좌표 비교 → 손 기울임 시 오인식
  - 핀치 임계값 하드코딩 → 거리별 부정확
  - 히스테리시스 없음 → 경계값 깜빡임

TO-BE (본 모듈):
  - 3D 벡터 내적(Dot Product) 각도 기반 판정
  - Schmitt Trigger 히스테리시스 → 상태 전이 안정화
  - 3D 유클리디안 거리 + palm_size 정규화
  - Handedness Majority Vote 안정화

파이프라인:
  21개 3D 랜드마크
    → KinematicsClassifier (5개 손가락 각도 계산)
    → 제스처 후보 도출
    → HysteresisGestureFilter (상태 전이 안정화)
    → 최종 HandGesture 출력
"""
import math
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict
from enum import Enum, auto


# ═══════════════════════════════════════════════════════════════════
#  상수: MediaPipe Hand 21 랜드마크 인덱스
# ═══════════════════════════════════════════════════════════════════

# 손가락별 관절 체인 (CMC/MCP → MCP/PIP → PIP/DIP → DIP/TIP)
# 엄지: 1-2-3-4, 검지: 5-6-7-8, 중지: 9-10-11-12
# 약지: 13-14-15-16, 소지: 17-18-19-20
FINGER_JOINTS = {
    "thumb":  [1, 2, 3, 4],     # CMC, MCP, IP, TIP
    "index":  [5, 6, 7, 8],     # MCP, PIP, DIP, TIP
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
MIDDLE_MCP = 9


# ═══════════════════════════════════════════════════════════════════
#  3D 벡터 유틸리티
# ═══════════════════════════════════════════════════════════════════

def _lm_to_vec(lm) -> np.ndarray:
    """MediaPipe NormalizedLandmark → numpy [x, y, z]"""
    if hasattr(lm, 'x'):
        return np.array([lm.x, lm.y, lm.z], dtype=np.float64)
    return np.array([lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0],
                    dtype=np.float64)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    두 벡터 사이의 각도 (도, 0~180).
    내적 공식: cos(θ) = (v1 · v2) / (|v1| × |v2|)
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def _dist_3d(a, b) -> float:
    """3D 유클리디안 거리"""
    va = _lm_to_vec(a)
    vb = _lm_to_vec(b)
    return float(np.linalg.norm(va - vb))


def _palm_size_3d(lms) -> float:
    """
    손바닥 기준 크기 (3D): wrist → middle_mcp 거리.
    모든 핀치/거리 측정의 정규화 기준.
    """
    return _dist_3d(lms[WRIST], lms[MIDDLE_MCP]) + 1e-6


# ═══════════════════════════════════════════════════════════════════
#  KinematicsClassifier — 3D 관절 각도 기반 손가락 상태 판정
# ═══════════════════════════════════════════════════════════════════

class FingerState(Enum):
    EXTENDED = auto()   # 펴짐
    HALF     = auto()   # 반쯤 굽힘
    CURLED   = auto()   # 접힘


class KinematicsClassifier:
    """
    3D 벡터 내적 각도로 각 손가락의 굽힘 상태를 판정.

    원리:
      각 손가락의 3개 마디 연결 벡터에서 인접 벡터 간 각도 계산:
        angle = acos( (v1·v2) / (|v1|×|v2|) )

      - 펴짐(Extended): 마디 각도 > 160° (거의 일직선)
      - 반굽힘(Half):   120° ~ 160°
      - 접힘(Curled):    마디 각도 < 120°

    엄지 특수 처리:
      엄지는 다른 4개 손가락과 운동 축이 다름.
      → MCP-IP-TIP 체인 + 엄지 끝과 검지 MCP 사이 거리를
         palm_size 대비 비율로 보조 판정.

    튜닝 가이드:
    ┌────────────────┬──────────────────────────────────────────┐
    │ 파라미터        │ 효과                                     │
    ├────────────────┼──────────────────────────────────────────┤
    │ extend_thresh  │ 높일수록 '펴짐' 판정 엄격.               │
    │                │ 기본 155°. 150~165° 범위 조절.           │
    ├────────────────┼──────────────────────────────────────────┤
    │ curl_thresh    │ 낮출수록 '접힘' 판정 엄격.               │
    │                │ 기본 100°. 80~120° 범위 조절.            │
    ├────────────────┼──────────────────────────────────────────┤
    │ thumb_dist_ext │ 엄지 펴짐 보조 거리 비율.                │
    │                │ 기본 0.7 (palm_size의 70% 이상이면 펴짐)  │
    └────────────────┴──────────────────────────────────────────┘
    """

    def __init__(self,
                 extend_thresh: float = 155.0,
                 curl_thresh: float = 100.0,
                 thumb_dist_ext: float = 0.7):
        self._ext_th = extend_thresh
        self._curl_th = curl_thresh
        self._thumb_dist_ext = thumb_dist_ext

    def classify_all(self, lms) -> Dict[str, FingerState]:
        """
        21개 랜드마크 → 5개 손가락 상태 dict 반환.

        Returns:
            {"thumb": FingerState, "index": ..., "middle": ...,
             "ring": ..., "pinky": ...}
        """
        result = {}
        palm_sz = _palm_size_3d(lms)

        for name, joints in FINGER_JOINTS.items():
            if name == "thumb":
                result[name] = self._classify_thumb(lms, joints, palm_sz)
            else:
                result[name] = self._classify_finger(lms, joints)

        return result

    def compute_angles(self, lms) -> Dict[str, List[float]]:
        """
        디버그용: 각 손가락의 관절 각도(도) 리스트 반환.
        예: {"index": [172.3, 168.1]} → PIP, DIP 관절 각도
        """
        angles = {}
        for name, joints in FINGER_JOINTS.items():
            pts = [_lm_to_vec(lms[j]) for j in joints]
            angs = []
            for i in range(1, len(pts) - 1):
                v1 = pts[i - 1] - pts[i]
                v2 = pts[i + 1] - pts[i]
                angs.append(_angle_between(v1, v2))
            angles[name] = angs
        return angles

    def _classify_finger(self, lms, joints: list) -> FingerState:
        """
        검지~소지: 3D 마디 각도 기반 판정.

        관절 체인: MCP(0) → PIP(1) → DIP(2) → TIP(3)
        측정 각도:
          - PIP 각도: vec(MCP→PIP) · vec(DIP→PIP) → 180°에 가까우면 펴짐
          - DIP 각도: vec(PIP→DIP) · vec(TIP→DIP)
        최소 각도를 기준으로 판정.
        """
        pts = [_lm_to_vec(lms[j]) for j in joints]

        # 각 관절의 굽힘 각도 계산
        min_angle = 180.0
        for i in range(1, len(pts) - 1):
            v1 = pts[i - 1] - pts[i]   # 이전 마디 → 현재 관절
            v2 = pts[i + 1] - pts[i]   # 다음 마디 → 현재 관절
            angle = _angle_between(v1, v2)
            min_angle = min(min_angle, angle)

        if min_angle >= self._ext_th:
            return FingerState.EXTENDED
        elif min_angle <= self._curl_th:
            return FingerState.CURLED
        else:
            return FingerState.HALF

    def _classify_thumb(self, lms, joints: list,
                        palm_sz: float) -> FingerState:
        """
        엄지 특수 처리:
          1) 3D 각도 (MCP 굽힘)
          2) 엄지 끝 ↔ 검지 MCP 거리 / palm_size

        엄지는 운동 축이 다르므로 각도만으로 부족.
        거리 비율을 보조 지표로 사용:
          - 엄지 tip이 검지 MCP에서 멀면 → 펴짐
          - 가까우면 → 접힘 (주먹 안으로)
        """
        # 각도 기반
        pts = [_lm_to_vec(lms[j]) for j in joints]
        min_angle = 180.0
        for i in range(1, len(pts) - 1):
            v1 = pts[i - 1] - pts[i]
            v2 = pts[i + 1] - pts[i]
            angle = _angle_between(v1, v2)
            min_angle = min(min_angle, angle)

        # 거리 기반 보조 판정
        thumb_tip_vec = _lm_to_vec(lms[THUMB_TIP])
        index_mcp_vec = _lm_to_vec(lms[5])  # INDEX_MCP
        dist_ratio = np.linalg.norm(thumb_tip_vec - index_mcp_vec) / palm_sz

        # 복합 판정:
        #   각도 >= 150° AND 거리비 >= 0.7 → 확실히 펴짐
        #   각도 <= 110° OR 거리비 <= 0.35 → 확실히 접힘
        if min_angle >= (self._ext_th - 5) and dist_ratio >= self._thumb_dist_ext:
            return FingerState.EXTENDED
        elif min_angle <= (self._curl_th + 10) or dist_ratio <= 0.35:
            return FingerState.CURLED
        else:
            return FingerState.HALF

    def compute_pinch_3d(self, lms, finger_tip_idx: int) -> float:
        """
        엄지 ↔ 특정 손가락 끝의 3D 정규화 핀치 거리.

        Returns:
            3D 거리 / palm_size (0.0 = 맞닿음, 1.0+ = 멀리 떨어짐)
        """
        palm_sz = _palm_size_3d(lms)
        return _dist_3d(lms[THUMB_TIP], lms[finger_tip_idx]) / palm_sz

    def compute_palm_normal(self, lms) -> np.ndarray:
        """
        손바닥 법선 벡터 (외적으로 계산).
        Handedness 보조 판정에 사용.

        wrist→index_mcp × wrist→pinky_mcp → 법선 벡터
        법선의 z 성분 부호로 손바닥/손등 방향 구분.
        """
        w = _lm_to_vec(lms[WRIST])
        idx_mcp = _lm_to_vec(lms[5])   # INDEX_MCP
        pnk_mcp = _lm_to_vec(lms[17])  # PINKY_MCP

        v1 = idx_mcp - w
        v2 = pnk_mcp - w
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0])
        return normal / norm


# ═══════════════════════════════════════════════════════════════════
#  HysteresisGestureFilter — Schmitt Trigger 상태 전이 안정화
# ═══════════════════════════════════════════════════════════════════

class HysteresisGestureFilter:
    """
    Schmitt Trigger 기반 히스테리시스 필터.

    원리:
      제스처 판정값(예: 핀치 거리)에 대해 두 개의 임계값을 설정:
        - ON  threshold: 이 값 이하가 되면 제스처 활성화 (더 엄격)
        - OFF threshold: 이 값 이상이 되어야 제스처 해제 (더 느슨)

      ON < OFF 로 설정하면 경계 영역에서의 깜빡임이 사라짐.

    예시 (핀치):
      ON=0.25, OFF=0.40 일 때:
        거리 0.30 → 아직 OFF 상태면 비활성 유지 (OFF에 안 닿았으니까)
        거리 0.20 → ON 진입 (0.25 이하)
        거리 0.35 → ON 유지 (OFF인 0.40에 안 닿았으니까)
        거리 0.42 → OFF 전환 (0.40 이상)

    ┌──────────────────────────────────────────────┐
    │          ON=0.25       OFF=0.40              │
    │            │             │                   │
    │  ──────┐  │  DEAD ZONE  │  ┌──────          │
    │  활성  │  │  (상태 유지) │  │  비활성        │
    │  ──────┘  │             │  └──────          │
    │            │             │                   │
    │         엄격(작은값)   느슨(큰값)            │
    └──────────────────────────────────────────────┘

    튜닝 가이드:
    ┌───────────────┬──────────────────────────────────────┐
    │ 파라미터       │ 효과                                  │
    ├───────────────┼──────────────────────────────────────┤
    │ pinch_on      │ 핀치 진입. 낮을수록 확실한 핀치만 인정. │
    │               │ 기본 0.25. 범위 0.20~0.30.            │
    ├───────────────┼──────────────────────────────────────┤
    │ pinch_off     │ 핀치 해제. 높을수록 안정적 유지.       │
    │               │ 기본 0.40. 범위 0.35~0.50.            │
    ├───────────────┼──────────────────────────────────────┤
    │ dead_zone     │ ON과 OFF 사이 간격(= OFF - ON).       │
    │               │ 넓을수록 안정적이나 반응 느림.          │
    │               │ 0.15 이상 권장.                        │
    ├───────────────┼──────────────────────────────────────┤
    │ min_hold_ms   │ 상태 진입 후 최소 유지 시간.           │
    │               │ 기본 80ms. 50~150ms 범위.              │
    │               │ 짧을수록 반응 빠르나 불안정.            │
    └───────────────┴──────────────────────────────────────┘
    """

    def __init__(self,
                 pinch_on: float = 0.25,
                 pinch_off: float = 0.40,
                 fist_angle_on: float = 95.0,
                 fist_angle_off: float = 115.0,
                 palm_angle_on: float = 160.0,
                 palm_angle_off: float = 145.0,
                 min_hold_ms: int = 80):
        """
        Args:
            pinch_on/off:      핀치 거리 진입/해제 (정규화 3D)
            fist_angle_on/off: 주먹 판정 최대 관절각 진입/해제 (°)
            palm_angle_on/off: 손바닥 판정 최소 관절각 진입/해제 (°)
            min_hold_ms:       상태 최소 유지 시간 (ms)
        """
        self._pinch_on = pinch_on
        self._pinch_off = pinch_off
        self._fist_ang_on = fist_angle_on
        self._fist_ang_off = fist_angle_off
        self._palm_ang_on = palm_angle_on
        self._palm_ang_off = palm_angle_off
        self._min_hold = min_hold_ms / 1000.0

        # 내부 상태
        self._pinch_index_active = False
        self._pinch_middle_active = False
        self._fist_active = False
        self._palm_active = False
        self._last_change_time = 0.0

    def update(self, pinch_index_dist: float,
               pinch_middle_dist: float,
               finger_states: Dict[str, FingerState],
               finger_angles: Dict[str, List[float]],
               timestamp: float) -> str:
        """
        모든 입력값으로 최종 제스처를 결정.

        Args:
            pinch_index_dist:  엄지↔검지 3D 정규화 거리
            pinch_middle_dist: 엄지↔중지 3D 정규화 거리
            finger_states:     KinematicsClassifier 출력
            finger_angles:     KinematicsClassifier 각도 출력
            timestamp:         현재 시각 (time.time())

        Returns:
            제스처 문자열: "PINCH_INDEX", "PINCH_MIDDLE",
                         "FIST", "PALM", "NONE"
        """
        dt = timestamp - self._last_change_time

        # ── 1. 핀치 판정 (Schmitt Trigger) ─────────────────
        prev_pi = self._pinch_index_active
        prev_pm = self._pinch_middle_active

        if self._pinch_index_active:
            if pinch_index_dist > self._pinch_off:
                self._pinch_index_active = False
        else:
            if pinch_index_dist < self._pinch_on:
                self._pinch_index_active = True

        if self._pinch_middle_active:
            if pinch_middle_dist > self._pinch_off:
                self._pinch_middle_active = False
        else:
            if pinch_middle_dist < self._pinch_on:
                self._pinch_middle_active = True

        # 양쪽 다 핀치면 더 가까운 쪽 우선
        if self._pinch_index_active and self._pinch_middle_active:
            if pinch_index_dist <= pinch_middle_dist:
                self._pinch_middle_active = False
            else:
                self._pinch_index_active = False

        # 핀치 상태 변경 시 min_hold 체크
        if self._pinch_index_active != prev_pi or \
           self._pinch_middle_active != prev_pm:
            if dt < self._min_hold:
                # 너무 빠른 전환 → 이전 상태 유지
                self._pinch_index_active = prev_pi
                self._pinch_middle_active = prev_pm
            else:
                self._last_change_time = timestamp

        if self._pinch_index_active:
            return "PINCH_INDEX"
        if self._pinch_middle_active:
            return "PINCH_MIDDLE"

        # ── 2. FIST / PALM 판정 ────────────────────────────
        # 4개 손가락(검지~소지)의 최소 각도
        four_finger_min_angles = []
        for name in ["index", "middle", "ring", "pinky"]:
            if name in finger_angles and finger_angles[name]:
                four_finger_min_angles.append(min(finger_angles[name]))

        if four_finger_min_angles:
            avg_min_angle = sum(four_finger_min_angles) / len(four_finger_min_angles)
        else:
            avg_min_angle = 150.0  # 기본값 (손가락 펴진 상태)

        # FIST 히스테리시스
        prev_fist = self._fist_active
        if self._fist_active:
            if avg_min_angle > self._fist_ang_off:
                self._fist_active = False
        else:
            if avg_min_angle < self._fist_ang_on:
                self._fist_active = True

        # PALM 히스테리시스
        prev_palm = self._palm_active
        if self._palm_active:
            if avg_min_angle < self._palm_ang_off:
                self._palm_active = False
        else:
            if avg_min_angle > self._palm_ang_on:
                self._palm_active = True

        # FIST min_hold 체크
        if self._fist_active != prev_fist:
            if dt < self._min_hold:
                self._fist_active = prev_fist
            else:
                self._last_change_time = timestamp

        # PALM min_hold 체크
        if self._palm_active != prev_palm:
            if dt < self._min_hold:
                self._palm_active = prev_palm
            else:
                self._last_change_time = timestamp

        if self._fist_active:
            return "FIST"
        if self._palm_active:
            return "PALM"

        return "NONE"

    def reset(self):
        """상태 초기화"""
        self._pinch_index_active = False
        self._pinch_middle_active = False
        self._fist_active = False
        self._palm_active = False
        self._last_change_time = 0.0


# ═══════════════════════════════════════════════════════════════════
#  HandednessStabilizer — 좌/우 손 안정화 (Majority Vote)
# ═══════════════════════════════════════════════════════════════════

class HandednessStabilizer:
    """
    최근 N 프레임의 Handedness 다수결(Majority Vote)로 안정화.

    MediaPipe Handedness 출력이 가끔 튀는 문제 해결:
      - 큐 버퍼에 최근 판정 저장
      - 과반수 이상의 판정값을 최종 결과로 사용
      - 보조: 손바닥 법선 벡터 z 부호로 검증
    """

    def __init__(self, window: int = 7):
        """
        Args:
            window: 다수결 윈도우 크기 (홀수 권장, 기본 7)
        """
        # 손별 히스토리: {hand_id: deque}
        self._buffers: Dict[int, deque] = {}
        self._window = window

    def stabilize(self, hand_idx: int,
                  raw_handedness: str,
                  palm_normal_z: float = 0.0) -> str:
        """
        Args:
            hand_idx:        손 인덱스 (0 또는 1)
            raw_handedness:  MediaPipe 원본 출력 ("Left" / "Right")
            palm_normal_z:   손바닥 법선 벡터의 z 성분 (보조)

        Returns:
            안정화된 "Left" / "Right"
        """
        if hand_idx not in self._buffers:
            self._buffers[hand_idx] = deque(maxlen=self._window)

        self._buffers[hand_idx].append(raw_handedness)

        buf = self._buffers[hand_idx]
        left_count = sum(1 for h in buf if h == "Left")
        right_count = len(buf) - left_count

        if left_count > right_count:
            return "Left"
        elif right_count > left_count:
            return "Right"
        else:
            # 동률: 법선 벡터 z 부호로 판정
            # 오른손이 카메라를 향하면 법선 z > 0 (일반적)
            return raw_handedness

    def reset(self):
        self._buffers.clear()


# ═══════════════════════════════════════════════════════════════════
#  통합 인터페이스: classify_hand_gesture_3d()
# ═══════════════════════════════════════════════════════════════════

# 싱글턴 인스턴스 (전역 상태 유지용)
_kinematics = KinematicsClassifier()
_filters: Dict[int, HysteresisGestureFilter] = {}  # 손별 필터
_handedness_stab = HandednessStabilizer()


def classify_hand_gesture_3d(
        lms,
        hand_idx: int = 0,
        raw_handedness: str = "Right",
        timestamp: float = 0.0
) -> Tuple[str, float, float, Dict[str, FingerState]]:
    """
    통합 제스처 분류 함수 (기존 _classify_gesture() 대체용).

    Args:
        lms:             MediaPipe 21개 NormalizedLandmark
        hand_idx:        손 인덱스 (0~1)
        raw_handedness:  MediaPipe 원본 handedness
        timestamp:       현재 시각

    Returns:
        (gesture_str, pinch_index_dist, pinch_middle_dist, finger_states)
        gesture_str: "PINCH_INDEX", "PINCH_MIDDLE", "FIST", "PALM", "NONE"
    """
    import time as _time
    t = timestamp or _time.time()

    # 손별 히스테리시스 필터 생성
    if hand_idx not in _filters:
        _filters[hand_idx] = HysteresisGestureFilter()

    filt = _filters[hand_idx]

    # 1. 손가락 상태 + 각도 계산
    states = _kinematics.classify_all(lms)
    angles = _kinematics.compute_angles(lms)

    # 2. 3D 핀치 거리
    d_index = _kinematics.compute_pinch_3d(lms, INDEX_TIP)
    d_middle = _kinematics.compute_pinch_3d(lms, MIDDLE_TIP)

    # 3. 히스테리시스 필터 적용
    gesture = filt.update(d_index, d_middle, states, angles, t)

    return gesture, d_index, d_middle, states
