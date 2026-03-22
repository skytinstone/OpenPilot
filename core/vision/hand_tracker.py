"""
손 추적 모듈 — MediaPipe HandLandmarker (Tasks API v0.10+)

21개 랜드마크로 손 제스처를 분류:
  PALM        — 손바닥 펼침 (트래킹 활성 상태)
  PINCH_INDEX — 엄지 + 검지 핀치 → 좌클릭
  PINCH_MIDDLE— 엄지 + 중지 핀치 → 우클릭
  FIST        — 주먹 → 스크롤
  NONE        — 감지 없음

특수 제스처 감지기:
  PalmRubDetector — 양 손바닥을 맞댄 채 문지르기 → 창 닫기
"""
import cv2
import numpy as np
import os
import time
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
            num_hands=cfg.get("max_num_hands", 2),
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

    def process_multi(self, frame: np.ndarray) -> List[HandData]:
        """
        BGR 프레임 → 감지된 모든 손의 HandData 리스트 반환 (최대 2개)
        각 HandData의 handedness 필드로 "Left" / "Right" 구분 가능
        """
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        hands: List[HandData] = []
        for i, lms in enumerate(result.hand_landmarks):
            handedness = (
                result.handedness[i][0].display_name
                if result.handedness else "Right"
            )
            gesture, d_idx, d_mid = _classify_gesture(lms, handedness)
            cx = (lms[WRIST].x + lms[MIDDLE_MCP].x) / 2
            cy = (lms[WRIST].y + lms[MIDDLE_MCP].y) / 2
            hands.append(HandData(
                landmarks=lms,
                gesture=gesture,
                hand_center=(cx, cy),
                pinch_index_dist=d_idx,
                pinch_middle_dist=d_mid,
                handedness=handedness,
            ))
        return hands

    def draw_debug(self, frame: np.ndarray, hand_data: Optional[HandData]) -> np.ndarray:
        """
        손 뼈대 + 제스처 시각화

        - 손가락별 고유 색상으로 뼈대(skeleton) 표시
        - 관절(joint)은 크기로 구분: 손끝 > MCP > 기타
        - 핀치 거리: 두 손가락 끝 사이 원형 시각화
        - 제스처 이름을 손목 옆에 표시
        """
        if hand_data is None:
            return frame

        h, w = frame.shape[:2]
        lms = hand_data.landmarks

        def px(lm):
            return int(lm.x * w), int(lm.y * h)

        # ── 손가락별 색상 (BGR) ───────────────────────────────
        FINGER_COLORS = {
            "thumb":  (0,   220, 255),  # 노랑  — 엄지
            "index":  (80,  255, 80),   # 초록  — 검지
            "middle": (255, 200, 0),    # 하늘  — 중지
            "ring":   (255, 80,  200),  # 보라  — 약지
            "pinky":  (80,  80,  255),  # 파랑  — 소지
            "palm":   (160, 160, 160),  # 회색  — 손바닥
        }

        # ── 손가락별 연결선 정의 ──────────────────────────────
        FINGER_BONES = {
            "thumb":  [(0,1),(1,2),(2,3),(3,4)],
            "index":  [(0,5),(5,6),(6,7),(7,8)],
            "middle": [(0,9),(9,10),(10,11),(11,12)],
            "ring":   [(0,13),(13,14),(14,15),(15,16)],
            "pinky":  [(0,17),(17,18),(18,19),(19,20)],
            "palm":   [(5,9),(9,13),(13,17),(0,17)],
        }

        # 손가락 끝 인덱스 (큰 원으로 표시)
        FINGERTIPS = {4, 8, 12, 16, 20}
        # MCP 관절 인덱스 (중간 원)
        MCPS = {1, 5, 9, 13, 17}

        # 관절 → 색상 매핑
        LM_COLORS = {}
        for finger, bones in FINGER_BONES.items():
            color = FINGER_COLORS[finger]
            for a, b in bones:
                LM_COLORS[a] = color
                LM_COLORS[b] = color

        # ── 뼈대 그리기 ───────────────────────────────────────
        for finger, bones in FINGER_BONES.items():
            color = FINGER_COLORS[finger]
            for a, b in bones:
                cv2.line(frame, px(lms[a]), px(lms[b]), color, 2, cv2.LINE_AA)

        # ── 관절 원 그리기 ────────────────────────────────────
        for i, lm in enumerate(lms):
            c = LM_COLORS.get(i, FINGER_COLORS["palm"])
            p = px(lm)
            if i in FINGERTIPS:
                # 손끝: 큰 원 + 흰 테두리
                cv2.circle(frame, p, 9, c, -1, cv2.LINE_AA)
                cv2.circle(frame, p, 9, (255, 255, 255), 1, cv2.LINE_AA)
            elif i in MCPS:
                # MCP: 중간 원
                cv2.circle(frame, p, 6, c, -1, cv2.LINE_AA)
                cv2.circle(frame, p, 6, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                # 나머지: 작은 원
                cv2.circle(frame, p, 4, c, -1, cv2.LINE_AA)

        # ── 핀치 거리 시각화 ─────────────────────────────────
        PINCH_THRESH = 0.35
        thumb_px = px(lms[THUMB_TIP])

        # 엄지 ↔ 검지 (초록)
        idx_px = px(lms[INDEX_TIP])
        mid_pt_i = ((thumb_px[0]+idx_px[0])//2, (thumb_px[1]+idx_px[1])//2)
        dist_i = hand_data.pinch_index_dist
        pinch_i_color = (0, 255, 80) if dist_i < PINCH_THRESH else (80, 80, 80)
        cv2.line(frame, thumb_px, idx_px, pinch_i_color, 1, cv2.LINE_AA)
        radius_i = max(4, int(dist_i * 60))
        cv2.circle(frame, mid_pt_i, radius_i, pinch_i_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"L:{dist_i:.2f}", (mid_pt_i[0]+6, mid_pt_i[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, pinch_i_color, 1)

        # 엄지 ↔ 중지 (빨강)
        mid_px = px(lms[MIDDLE_TIP])
        mid_pt_m = ((thumb_px[0]+mid_px[0])//2, (thumb_px[1]+mid_px[1])//2)
        dist_m = hand_data.pinch_middle_dist
        pinch_m_color = (0, 80, 255) if dist_m < PINCH_THRESH else (80, 80, 80)
        cv2.line(frame, thumb_px, mid_px, pinch_m_color, 1, cv2.LINE_AA)
        radius_m = max(4, int(dist_m * 60))
        cv2.circle(frame, mid_pt_m, radius_m, pinch_m_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"R:{dist_m:.2f}", (mid_pt_m[0]+6, mid_pt_m[1]+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, pinch_m_color, 1)

        # ── 제스처 이름 (손목 옆) ─────────────────────────────
        GESTURE_LABELS = {
            HandGesture.PALM:         "PALM",
            HandGesture.PINCH_INDEX:  "LEFT CLICK",
            HandGesture.PINCH_MIDDLE: "RIGHT CLICK",
            HandGesture.FIST:         "SCROLL",
            HandGesture.NONE:         "NONE",
        }
        GESTURE_BADGE_COLORS = {
            HandGesture.PALM:         (80,  220, 80),
            HandGesture.PINCH_INDEX:  (80,  80,  255),
            HandGesture.PINCH_MIDDLE: (80,  80,  255),
            HandGesture.FIST:         (0,   165, 255),
            HandGesture.NONE:         (120, 120, 120),
        }
        g_label = GESTURE_LABELS.get(hand_data.gesture, "")
        g_color = GESTURE_BADGE_COLORS.get(hand_data.gesture, (120, 120, 120))
        wrist_px = px(lms[WRIST])
        badge_pos = (wrist_px[0] - 10, wrist_px[1] + 24)

        # 배지 배경
        (tw, th), _ = cv2.getTextSize(g_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame,
                      (badge_pos[0] - 4, badge_pos[1] - th - 4),
                      (badge_pos[0] + tw + 4, badge_pos[1] + 4),
                      (20, 20, 20), -1)
        cv2.rectangle(frame,
                      (badge_pos[0] - 4, badge_pos[1] - th - 4),
                      (badge_pos[0] + tw + 4, badge_pos[1] + 4),
                      g_color, 1)
        cv2.putText(frame, g_label, badge_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, g_color, 2, cv2.LINE_AA)

        # ── 범례 (우측 상단) ──────────────────────────────────
        legend = [
            ("Thumb",  FINGER_COLORS["thumb"]),
            ("Index",  FINGER_COLORS["index"]),
            ("Middle", FINGER_COLORS["middle"]),
            ("Ring",   FINGER_COLORS["ring"]),
            ("Pinky",  FINGER_COLORS["pinky"]),
        ]
        lx, ly = w - 110, 50
        for name, color in legend:
            cv2.circle(frame, (lx, ly), 5, color, -1)
            cv2.putText(frame, name, (lx + 12, ly + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            ly += 20

        return frame

    def close(self):
        self._landmarker.close()
        print("[HandTracker] 종료됨")


# ── PalmRubDetector ──────────────────────────────────────────────

class PalmRubDetector:
    """
    양 손바닥을 서로 맞댄 채 좌우로 문지르는 제스처 감지.

    조건:
      1. 양손 모두 PALM 제스처
      2. 두 손 중심 거리 < proximity_thresh (손이 가까이 붙어있음)
      3. 상대 X 속도 방향이 rub_window 초 안에 rub_count 번 이상 바뀜

    반환: update() 가 True 이면 제스처 완성 → Cmd+W 실행 권장

    속성:
      progress  — 완성까지 진행률 0.0~1.0 (시각화용)
      is_active — 현재 손이 근접 + PALM 상태인지
    """

    def __init__(self,
                 proximity_thresh: float = 0.38,   # 정규화 손 중심 거리
                 rub_count: int = 4,                # 필요 방향 전환 횟수
                 rub_window: float = 1.8,           # 감지 시간 창 (초)
                 cooldown: float = 1.5):            # 트리거 후 쿨다운 (초)
        self._prox       = proximity_thresh
        self._rub_count  = rub_count
        self._window     = rub_window
        self._cooldown   = cooldown

        self._prev_rel_x    : Optional[float] = None
        self._prev_time     : Optional[float] = None
        self._prev_dir      : int             = 0
        self._changes       : List[float]     = []   # 방향 전환 타임스탬프
        self._last_trigger  : float           = 0.0

        self.progress  : float = 0.0   # 0~1 시각화용
        self.is_active : bool  = False  # 손 근접 + PALM 여부

    def update(self, left: Optional["HandData"],
               right: Optional["HandData"]) -> bool:
        """
        매 프레임 호출. 제스처 완성 시 True 반환.
        left / right: HandData 또는 None
        """
        now = time.time()

        # 쿨다운
        if now - self._last_trigger < self._cooldown:
            self.is_active = False
            return False

        # 양손 PALM 체크
        if (left is None or right is None
                or left.gesture  != HandGesture.PALM
                or right.gesture != HandGesture.PALM):
            self._reset()
            return False

        # 근접 체크 (정규화 유클리드 거리)
        lx, ly = left.hand_center
        rx, ry = right.hand_center
        dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
        if dist > self._prox:
            self._reset()
            return False

        self.is_active = True

        # 상대 X 위치 (왼손 - 오른손)
        rel_x = lx - rx

        if self._prev_rel_x is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.01:
                vel = (rel_x - self._prev_rel_x) / dt
                # 속도 임계값 0.15/s 이상일 때만 방향 인정
                direction = (1 if vel > 0.15 else -1 if vel < -0.15 else 0)

                if direction != 0 and direction != self._prev_dir:
                    self._changes.append(now)
                    self._prev_dir = direction

                # 시간 창 밖 항목 제거
                cutoff = now - self._window
                self._changes = [t for t in self._changes if t > cutoff]

                self.progress = min(len(self._changes) / self._rub_count, 1.0)

                if len(self._changes) >= self._rub_count:
                    self._changes.clear()
                    self._prev_dir    = 0
                    self._last_trigger = now
                    self.progress      = 0.0
                    self.is_active     = False
                    return True

        self._prev_rel_x = rel_x
        self._prev_time  = now
        return False

    def _reset(self):
        self._prev_rel_x = None
        self._prev_time  = None
        self._prev_dir   = 0
        self._changes    = []
        self.progress    = 0.0
        self.is_active   = False
