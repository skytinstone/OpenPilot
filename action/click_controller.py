"""
클릭 컨트롤러 — macOS CGEvent 기반 좌/우 클릭

현재 마우스 커서 위치에서 클릭을 실행합니다.
(커서 위치는 Phase 1 Eye Tracking 이 제어)
"""
import time


def _current_pos():
    """현재 마우스 커서 위치 반환"""
    from Quartz import CGEventCreate, CGEventGetLocation
    event = CGEventCreate(None)
    return CGEventGetLocation(event)


def _click(down_type, up_type, button):
    from Quartz import CGEventCreateMouseEvent, CGEventPost, kCGHIDEventTap
    pos = _current_pos()
    down = CGEventCreateMouseEvent(None, down_type, pos, button)
    up   = CGEventCreateMouseEvent(None, up_type,   pos, button)
    CGEventPost(kCGHIDEventTap, down)
    time.sleep(0.04)
    CGEventPost(kCGHIDEventTap, up)


class ClickController:
    """
    좌/우 클릭 with 쿨다운 (연속 오클릭 방지)
    cooldown_ms: 핀치 후 다음 클릭까지 최소 대기 시간 (ms)
    """
    def __init__(self, cooldown_ms: int = 600):
        self._cooldown = cooldown_ms / 1000.0
        self._last_left  = 0.0
        self._last_right = 0.0

    def left_click(self) -> bool:
        """좌클릭. 쿨다운 중이면 False 반환."""
        from Quartz import (kCGEventLeftMouseDown, kCGEventLeftMouseUp,
                            kCGMouseButtonLeft)
        now = time.time()
        if now - self._last_left < self._cooldown:
            return False
        self._last_left = now
        _click(kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGMouseButtonLeft)
        return True

    def right_click(self) -> bool:
        """우클릭. 쿨다운 중이면 False 반환."""
        from Quartz import (kCGEventRightMouseDown, kCGEventRightMouseUp,
                            kCGMouseButtonRight)
        now = time.time()
        if now - self._last_right < self._cooldown:
            return False
        self._last_right = now
        _click(kCGEventRightMouseDown, kCGEventRightMouseUp, kCGMouseButtonRight)
        return True
