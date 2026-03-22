"""
스크롤 컨트롤러 — 주먹 제스처 기반 스크롤 + 관성(inertia)

동작 방식:
  1. 주먹 제스처 중 → 손 이동량에 비례해 즉시 스크롤
  2. 주먹 해제 시 속도가 빠르면(> INERTIA_TRIGGER) → 관성 스크롤 시작
  3. 관성: 초기 속도에서 DECAY 배율로 감속, 멈출 때까지 자동 스크롤
"""
import time
import threading
from collections import deque


def _do_scroll(pixels: int):
    """macOS CGEvent 픽셀 단위 스크롤 (양수=위, 음수=아래)"""
    try:
        from Quartz import (CGEventCreateScrollWheelEvent, CGEventPost,
                            kCGScrollEventUnitPixel, kCGHIDEventTap)
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 1, pixels)
        CGEventPost(kCGHIDEventTap, event)
    except Exception:
        pass


class ScrollController:
    """
    주요 파라미터:
      sensitivity    : 손 이동량(정규화) → 픽셀 변환 배율
      inertia_trigger: 이 속도(정규화/s) 이상이면 관성 발동
      inertia_decay  : 관성 감속 배율 (틱당, 16ms 기준)
    """

    SENSITIVITY     = 3500   # 클수록 같은 손 움직임에 더 빠른 스크롤
    INERTIA_TRIGGER = 0.35   # 관성 발동 최소 속도 (정규화 단위/s)
    INERTIA_DECAY   = 0.88   # 틱당 감속 배율 (0.88 ≈ 1초 후 거의 정지)
    INERTIA_MIN_VEL = 0.008  # 이 속도 이하에서 관성 종료
    TICK_S          = 0.016  # 관성 틱 간격 (약 60fps)

    def __init__(self):
        self._last_y:    float | None = None
        self._last_t:    float | None = None
        self._vel_buf = deque(maxlen=8)   # 최근 프레임 속도 버퍼

        self._inertia_vel = 0.0
        self._lock = threading.Lock()

        t = threading.Thread(target=self._inertia_loop, daemon=True)
        t.start()

    # ── 주먹 활성 중 ────────────────────────────────────────────

    def fist_update(self, y_norm: float):
        """
        주먹 제스처가 활성화된 매 프레임 호출.
        y_norm: 손 중심의 정규화 y좌표 (0=화면 위, 1=화면 아래)
        """
        now = time.time()

        # 관성 스크롤 중단 (직접 조작 우선)
        with self._lock:
            self._inertia_vel = 0.0

        if self._last_y is not None and self._last_t is not None:
            dt = now - self._last_t
            if 0 < dt < 0.12:
                dy = self._last_y - y_norm   # 손이 위로 → 양수 → 위로 스크롤
                vel = dy / dt                # 정규화 단위/s

                # 직접 스크롤
                px = int(dy * self.SENSITIVITY)
                if abs(px) >= 1:
                    _do_scroll(px)

                self._vel_buf.append(vel)

        self._last_y = y_norm
        self._last_t = now

    # ── 주먹 해제 ───────────────────────────────────────────────

    def fist_release(self):
        """주먹 제스처가 끝날 때 호출 — 빠른 스와이프면 관성 발동"""
        if self._vel_buf:
            avg_vel = sum(self._vel_buf) / len(self._vel_buf)
            if abs(avg_vel) > self.INERTIA_TRIGGER:
                with self._lock:
                    self._inertia_vel = avg_vel

        self._last_y = None
        self._last_t = None
        self._vel_buf.clear()

    def stop_inertia(self):
        """관성 즉시 중단"""
        with self._lock:
            self._inertia_vel = 0.0

    # ── 관성 루프 (백그라운드 스레드) ───────────────────────────

    def _inertia_loop(self):
        while True:
            time.sleep(self.TICK_S)
            with self._lock:
                vel = self._inertia_vel
                if abs(vel) < self.INERTIA_MIN_VEL:
                    self._inertia_vel = 0.0
                    continue
                px = int(vel * self.SENSITIVITY * self.TICK_S)
                self._inertia_vel *= self.INERTIA_DECAY

            if abs(px) >= 1:
                _do_scroll(px)

    @property
    def inertia_velocity(self) -> float:
        with self._lock:
            return self._inertia_vel
