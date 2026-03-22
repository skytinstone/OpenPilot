"""
macOS 마우스 컨트롤러
PyObjC CGEvent를 통해 네이티브 수준 커서 이동 실행

사전 요구사항:
  시스템 설정 → 개인 정보 보호 및 보안 → 접근성
  → 실행 중인 Terminal/IDE 앱에 접근성 권한 부여 필요
"""
import time
from typing import Tuple, Optional


def _get_screen_size() -> Tuple[int, int]:
    """macOS 논리 해상도 반환 (Retina 대응)"""
    try:
        from AppKit import NSScreen
        screen = NSScreen.mainScreen()
        frame = screen.frame()
        return int(frame.size.width), int(frame.size.height)
    except Exception:
        # fallback
        return 1440, 900


def check_accessibility() -> bool:
    """접근성 권한 확인"""
    try:
        from Quartz import CGPreflightPostEventAccess
        has_access = CGPreflightPostEventAccess()
        if not has_access:
            print("[ERROR] 접근성 권한이 없습니다.")
            print("       시스템 설정 → 개인 정보 보호 및 보안 → 접근성")
            print("       → 현재 터미널/IDE 앱을 추가하고 토글을 켜주세요.")
        return has_access
    except ImportError:
        print("[WARN] PyObjC가 설치되지 않았습니다. pip install pyobjc-framework-Quartz")
        return False


class MouseController:
    def __init__(self):
        self.screen_w, self.screen_h = _get_screen_size()
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._accessible = check_accessibility()
        print(f"[MouseController] 화면 해상도: {self.screen_w}x{self.screen_h}")

    def move(self, x: int, y: int):
        """커서를 (x, y)로 이동"""
        if not self._accessible:
            return

        # 화면 경계 클램핑
        x = max(0, min(x, self.screen_w - 1))
        y = max(0, min(y, self.screen_h - 1))

        try:
            from Quartz import (
                CGEventCreateMouseEvent,
                CGEventPost,
                kCGEventMouseMoved,
                kCGMouseButtonLeft,
                kCGHIDEventTap,
            )
            event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event)
            self._last_x = x
            self._last_y = y
        except Exception as e:
            print(f"[MouseController] 커서 이동 실패: {e}")

    def click(self, x: Optional[int] = None, y: Optional[int] = None):
        """현재 위치(또는 지정 위치)에서 좌클릭"""
        if not self._accessible:
            return

        cx = x if x is not None else (self._last_x or 0)
        cy = y if y is not None else (self._last_y or 0)

        try:
            from Quartz import (
                CGEventCreateMouseEvent,
                CGEventPost,
                kCGEventLeftMouseDown,
                kCGEventLeftMouseUp,
                kCGMouseButtonLeft,
                kCGHIDEventTap,
            )
            down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (cx, cy), kCGMouseButtonLeft)
            up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (cx, cy), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, down)
            time.sleep(0.05)
            CGEventPost(kCGHIDEventTap, up)
        except Exception as e:
            print(f"[MouseController] 클릭 실패: {e}")

    @property
    def position(self) -> Tuple[Optional[float], Optional[float]]:
        return self._last_x, self._last_y
