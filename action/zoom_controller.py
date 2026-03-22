"""
줌 컨트롤러 — 양손 핀치 거리 변화 → macOS Ctrl+Scroll 줌

동작:
  양손 검지+엄지 핀치 상태에서
  - 손이 멀어지면 → Zoom In
  - 손이 가까워지면 → Zoom Out

macOS는 Ctrl+Scroll 을 대부분의 앱(브라우저, 문서 등)에서 줌으로 처리함.
"""
import time
from typing import Tuple


def _ctrl_scroll(pixels: int):
    """Ctrl + 스크롤 이벤트 전송 (양수=줌인, 음수=줌아웃)"""
    try:
        from Quartz import (
            CGEventCreateScrollWheelEvent, CGEventSetFlags, CGEventPost,
            kCGScrollEventUnitPixel, kCGHIDEventTap, kCGEventFlagMaskControl,
        )
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 1, pixels)
        CGEventSetFlags(event, kCGEventFlagMaskControl)
        CGEventPost(kCGHIDEventTap, event)
    except Exception:
        pass


class ZoomController:
    """
    파라미터:
      sensitivity   : 정규화 거리 변화 → 스크롤 픽셀 배율 (클수록 빠른 줌)
      min_delta     : 이 값 이하의 거리 변화는 무시 (떨림 방지)
    """

    SENSITIVITY = 1200
    MIN_DELTA   = 0.004

    def __init__(self):
        self._last_dist: float | None = None
        self._active = False

    # ── 핀치 위치 → 거리 ─────────────────────────────────────────

    @staticmethod
    def pinch_center(hand_data) -> Tuple[float, float]:
        """엄지+검지 핀치 중심점 반환 (정규화 0~1)"""
        lms = hand_data.landmarks
        from core.vision.hand_tracker import THUMB_TIP, INDEX_TIP
        return (
            (lms[THUMB_TIP].x + lms[INDEX_TIP].x) / 2,
            (lms[THUMB_TIP].y + lms[INDEX_TIP].y) / 2,
        )

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # ── 상태 관리 ────────────────────────────────────────────────

    def start(self, left_hand, right_hand):
        """양손 핀치 진입 시 호출 — 초기 거리 기록"""
        lc = self.pinch_center(left_hand)
        rc = self.pinch_center(right_hand)
        self._last_dist = self._dist(lc, rc)
        self._active = True

    def update(self, left_hand, right_hand) -> float:
        """
        양손 핀치 유지 중 매 프레임 호출.
        줌 이벤트를 직접 전송하고, 방향 값 반환 (양수=줌인, 음수=줌아웃, 0=변화없음)
        """
        lc = self.pinch_center(left_hand)
        rc = self.pinch_center(right_hand)
        dist = self._dist(lc, rc)

        if self._last_dist is None:
            self._last_dist = dist
            return 0.0

        delta = dist - self._last_dist   # 양수 = 멀어짐 = 줌인
        self._last_dist = dist

        if abs(delta) < self.MIN_DELTA:
            return 0.0

        px = int(delta * self.SENSITIVITY)
        if abs(px) >= 1:
            _ctrl_scroll(px)

        return delta

    def stop(self):
        """양손 핀치 해제 시 호출"""
        self._last_dist = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active
