"""
실제 화면 위에 손 제스처 이벤트를 토스트 알림으로 표시하는 NSWindow 오버레이

- 제스처 발생 시 화면 하단 중앙에 배지가 나타났다 사라짐
- 종류별 색상 구분 (클릭=파랑, 스크롤=주황, 줌=초록)

사용법:
    overlay = GestureStatusOverlay()
    overlay.start()                      # 메인 스레드에서 호출
    overlay.show("LEFT CLICK", "click")  # 이벤트 표시
    overlay.refresh()                    # 매 프레임 호출 (fade-out 갱신)
    overlay.stop()
"""
import threading
import time


# ── 이벤트 종류별 색상 (R, G, B) ─────────────────────────────────
_EVENT_COLORS = {
    "click":  (0.25, 0.55, 1.0),    # 파랑
    "scroll": (1.0,  0.55, 0.10),   # 주황
    "zoom":   (0.15, 0.90, 0.45),   # 초록
    "default":(0.75, 0.75, 0.75),   # 회색
}


class _GestureEventState:
    def __init__(self):
        self._lock    = threading.Lock()
        self._msg     = ""
        self._color   = _EVENT_COLORS["default"]
        self._expire  = 0.0

    def show(self, msg: str, kind: str = "default", duration: float = 0.75):
        color = _EVENT_COLORS.get(kind, _EVENT_COLORS["default"])
        with self._lock:
            self._msg    = msg
            self._color  = color
            self._expire = time.time() + duration

    def get(self):
        with self._lock:
            if time.time() < self._expire:
                return self._msg, self._color, self._expire
            return "", _EVENT_COLORS["default"], 0.0


_gesture_state = _GestureEventState()


def _make_gesture_view_class(screen_w: int, screen_h: int):
    try:
        import objc
        from AppKit import NSView

        class GestureViewImpl(NSView):

            def initWithFrame_(self, frame):
                self = objc.super(GestureViewImpl, self).initWithFrame_(frame)
                if self is None:
                    return None
                self._sw = screen_w
                self._sh = screen_h
                return self

            def drawRect_(self, rect):
                msg, color, expire = _gesture_state.get()
                if not msg:
                    return
                try:
                    # 남은 시간으로 alpha 계산 (마지막 0.25초 동안 fade-out)
                    remaining = max(0.0, expire - time.time())
                    alpha = min(1.0, remaining / 0.25)
                    self._draw_badge(msg, color, alpha)
                except Exception:
                    pass

            def _draw_badge(self, msg: str, color, alpha: float):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                sw, sh = self._sw, self._sh
                r, g, b = color

                font     = NSFont.boldSystemFontOfSize_(22.0)
                ns_str   = NSString.stringWithString_(msg)
                attrs    = {
                    NSFontAttributeName:            font,
                    NSForegroundColorAttributeName: NSColor.whiteColor(),
                }
                attr_str = NSAttributedString.alloc().initWithString_attributes_(
                    ns_str, attrs
                )
                ts = attr_str.size()

                pad    = 18.0
                bw     = ts.width  + pad * 2
                bh     = ts.height + pad * 1.4
                bx     = (sw - bw) / 2
                by     = 60.0          # 화면 하단에서 60pt

                # 배경 (둥근 사각형)
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.08, 0.08, 0.08, 0.82 * alpha
                ).set()
                bg = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(bx, by, bw, bh), 12.0, 12.0
                )
                bg.fill()

                # 컬러 테두리
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    r, g, b, 0.9 * alpha
                ).set()
                bg.setLineWidth_(2.5)
                bg.stroke()

                # 텍스트
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, alpha
                ).set()
                # attrs 내부 색상은 고정이므로 alpha 반영된 컬러로 재생성
                attrs2 = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            1.0, 1.0, 1.0, alpha
                        ),
                }
                attr_str2 = NSAttributedString.alloc().initWithString_attributes_(
                    ns_str, attrs2
                )
                attr_str2.drawAtPoint_(
                    NSMakePoint(bx + pad, by + (bh - ts.height) / 2)
                )

        return GestureViewImpl

    except Exception as e:
        print(f"[GestureStatusOverlay] View 생성 실패: {e}")
        return None


class GestureStatusOverlay:
    """실제 화면 하단에 제스처 이벤트 토스트를 표시하는 NSWindow 오버레이"""

    def __init__(self):
        self._window = None
        self._view   = None
        self._active = False

    def start(self):
        """메인 스레드에서 호출"""
        try:
            from AppKit import (
                NSWindow, NSColor,
                NSBorderlessWindowMask, NSBackingStoreBuffered,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorStationary,
                NSScreen,
            )

            screen       = NSScreen.mainScreen()
            screen_frame = screen.frame()
            sw           = int(screen_frame.size.width)
            sh           = int(screen_frame.size.height)

            ViewClass = _make_gesture_view_class(sw, sh)
            if ViewClass is None:
                return

            self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                screen_frame,
                NSBorderlessWindowMask,
                NSBackingStoreBuffered,
                False,
            )
            self._window.setOpaque_(False)
            self._window.setBackgroundColor_(NSColor.clearColor())
            self._window.setLevel_(998)
            self._window.setIgnoresMouseEvents_(True)
            self._window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )

            self._view = ViewClass.alloc().initWithFrame_(screen_frame)
            self._window.setContentView_(self._view)
            self._window.makeKeyAndOrderFront_(None)
            self._active = True
            print("[GestureStatusOverlay] 제스처 상태 오버레이 시작")

        except ImportError:
            print("[GestureStatusOverlay] PyObjC 미설치 — 건너뜀")
        except Exception as e:
            print(f"[GestureStatusOverlay] 시작 실패: {e}")

    def show(self, msg: str, kind: str = "default", duration: float = 0.75):
        """제스처 이벤트 메시지 표시"""
        _gesture_state.show(msg, kind, duration)
        self.refresh()

    def refresh(self):
        """매 프레임 호출 — fade-out 갱신"""
        if self._view and self._active:
            try:
                self._view.setNeedsDisplay_(True)
            except Exception:
                pass

    def stop(self):
        if self._window and self._active:
            try:
                self._window.orderOut_(None)
            except Exception:
                pass
        self._active = False
        print("[GestureStatusOverlay] 제스처 상태 오버레이 종료")
