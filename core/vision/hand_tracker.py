"""
손 추적 모듈 — MediaPipe HandLandmarker (Tasks API v0.10+)

21개 랜드마크로 손 제스처를 분류:
  PALM        — 손바닥 펼침 (트래킹 활성 상태)
  PINCH_INDEX — 엄지 + 검지 핀치 → 좌클릭
  PINCH_MIDDLE— 엄지 + 중지 핀치 → 우클릭
  FIST        — 주먹 → 스크롤
  NONE        — 감지 없음
"""
import cv2
import numpy as np
import os
import urllib.request
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple

# ── 모델 자동 다운로드 ───────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("[HandTracker] 모델 파일 다운로드 중... (최초 1회)")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[HandTracker] 모델 다운로드 완료")


# ── 랜드마크 인덱스 ─────────────────────────────────────────────
WRIST          = 0
THUMB_TIP      = 4
THUMB_IP       = 3
INDEX_MCP      = 5
INDEX_PIP      = 6
INDEX_TIP      = 8
MIDDLE_MCP     = 9
MIDDLE_PIP     = 10
MIDDLE_TIP     = 12
RING_MCP       = 13
RING_PIP       = 14
RING_TIP       = 16
PINKY_MCP      = 17
PINKY_PIP      = 18
PINKY_TIP      = 20


class HandGesture(Enum):
    NONE          = auto()
    PALM          = auto()   # 손바닥 펼침
    PINCH_INDEX   = auto()   # 엄지+검지 핀치 → 좌클릭
    PINCH_MIDDLE  = auto()   # 엄지+중지 핀치 → 우클릭
    FIST          = auto()   # 주먹 → 스크롤


@dataclass
class HandData:
    landmarks:          list            # 21개 NormalizedLandmark
    gesture:            HandGesture
    hand_center:        Tuple[float, float]   # 정규화 (0~1)
    pinch_index_dist:   float           # 엄지+검지 정규화 거리
    pinch_middle_dist:  float           # 엄지+중지 정규화 거리
    handedness:         str             # "Left" / "Right"


# ── 제스처 인식 헬퍼 ─────────────────────────────────────────────

def _palm_size(lms) -> float:
    """손목 → 중지 MCP 거리 (정규화 기준)"""
    w, m = lms[WRIST], lms[MIDDLE_MCP]
    return ((w.x - m.x) ** 2 + (w.y - m.y) ** 2) ** 0.5 + 1e-6


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _finger_extended(lms, tip_idx, pip_idx) -> bool:
    """검지~소지: tip.y < pip.y 이면 펴진 상태 (y=0 이 위쪽)"""
    return lms[tip_idx].y < lms[pip_idx].y


def _finger_curled(lms, tip_idx, mcp_idx) -> bool:
    """tip.y > mcp.y 이면 접힌 상태"""
    return lms[tip_idx].y > lms[mcp_idx].y


def _classify_gesture(lms, handedness: str) -> tuple:
    """
    제스처 분류 + 핀치 거리 반환
    반환: (HandGesture, pinch_index_dist, pinch_middle_dist)
    """
    size = _palm_size(lms)

    # 핀치 거리 (정규화)
    d_index  = _dist(lms[THUMB_TIP], lms[INDEX_TIP])  / size
    d_middle = _dist(lms[THUMB_TIP], lms[MIDDLE_TIP]) / size

    PINCH_THRESH = 0.35   # 이 값 이하이면 핀치로 판정

    # ① PINCH_INDEX — 엄지+검지 핀치, 나머지 손가락 상태 무관
    if d_index < PINCH_THRESH and d_index < d_middle:
        return HandGesture.PINCH_INDEX, d_index, d_middle

    # ② PINCH_MIDDLE — 엄지+중지 핀치
    if d_middle < PINCH_THRESH and d_middle < d_index:
        return HandGesture.PINCH_MIDDLE, d_index, d_middle

    # ③ FIST — 4개 손가락 모두 접힘
    fingers_curled = all([
        _finger_curled(lms, INDEX_TIP,  INDEX_MCP),
        _finger_curled(lms, MIDDLE_TIP, MIDDLE_MCP),
        _finger_curled(lms, RING_TIP,   RING_MCP),
        _finger_curled(lms, PINKY_TIP,  PINKY_MCP),
    ])
    if fingers_curled:
        return HandGesture.FIST, d_index, d_middle

    # ④ PALM — 4개 손가락 모두 펴짐
    fingers_extended = all([
        _finger_extended(lms, INDEX_TIP,  INDEX_PIP),
        _finger_extended(lms, MIDDLE_TIP, MIDDLE_PIP),
        _finger_extended(lms, RING_TIP,   RING_PIP),
        _finger_extended(lms, PINKY_TIP,  PINKY_PIP),
    ])
    if fingers_extended:
        return HandGesture.PALM, d_index, d_middle

    return HandGesture.NONE, d_index, d_middle


# ── HandTracker 클래스 ───────────────────────────────────────────

class HandTracker:
    def __init__(self, config: dict = None):
        _ensure_model()

        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks import python as mp_python

        cfg = (config or {}).get("hand_tracking", {})

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            num_hands=cfg.get("max_num_hands", 1),
            min_hand_detection_confidence=cfg.get("min_detection_confidence", 0.6),
            min_hand_presence_confidence=cfg.get("min_presence_confidence", 0.6),
            min_tracking_confidence=cfg.get("min_tracking_confidence", 0.5),
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        print("[HandTracker] MediaPipe HandLandmarker 초기화 완료")

    def process(self, frame: np.ndarray) -> Optional[HandData]:
        """BGR 프레임 → HandData 반환 (손 미감지 시 None)"""
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        lms = result.hand_landmarks[0]
        handedness = (
            result.handedness[0][0].display_name
            if result.handedness else "Right"
        )

        gesture, d_idx, d_mid = _classify_gesture(lms, handedness)

        # 손 중심: 손목 + 중지 MCP 평균
        cx = (lms[WRIST].x + lms[MIDDLE_MCP].x) / 2
        cy = (lms[WRIST].y + lms[MIDDLE_MCP].y) / 2

        return HandData(
            landmarks=lms,
            gesture=gesture,
            hand_center=(cx, cy),
            pinch_index_dist=d_idx,
            pinch_middle_dist=d_mid,
            handedness=handedness,
        )

    def draw_debug(self, frame: np.ndarray, hand_data: Optional[HandData]) -> np.ndarray:
        """랜드마크 + 제스처 시각화"""
        if hand_data is None:
            return frame

        h, w = frame.shape[:2]
        lms = hand_data.landmarks

        def px(lm):
            return int(lm.x * w), int(lm.y * h)

        # 손가락 연결선
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),         # 엄지
            (0,5),(5,6),(6,7),(7,8),          # 검지
            (0,9),(9,10),(10,11),(11,12),     # 중지
            (0,13),(13,14),(14,15),(15,16),   # 약지
            (0,17),(17,18),(18,19),(19,20),   # 소지
            (5,9),(9,13),(13,17),             # 손바닥
        ]
        gesture_colors = {
            HandGesture.PALM:         (80, 220, 80),
            HandGesture.PINCH_INDEX:  (80, 80, 255),
            HandGesture.PINCH_MIDDLE: (255, 80, 80),
            HandGesture.FIST:         (0, 165, 255),
            HandGesture.NONE:         (120, 120, 120),
        }
        color = gesture_colors.get(hand_data.gesture, (120, 120, 120))

        for a, b in CONNECTIONS:
            cv2.line(frame, px(lms[a]), px(lms[b]), color, 2)

        for lm in lms:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)

        # 엄지-검지 핀치 거리 표시
        cv2.line(frame, px(lms[THUMB_TIP]), px(lms[INDEX_TIP]),  (80, 80, 255), 1)
        cv2.line(frame, px(lms[THUMB_TIP]), px(lms[MIDDLE_TIP]), (255, 80, 80), 1)

        return frame

    def close(self):
        self._landmarker.close()
        print("[HandTracker] 종료됨")
