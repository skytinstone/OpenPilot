"""
실제 화면 위에 시선 커서를 표시하는 투명 오버레이
— NSWindow 기반, 마우스 이벤트 무시

사용법:
    overlay = GazeCursorOverlay()
    overlay.start()          # 메인 스레드에서 호출
    overlay.update(x, y)     # 매 프레임 시선 좌표 업데이트
    overlay.stop()
"""
import threading
from typing import Optional


class GazeCursorState:
    """시선 좌표 스레드 안전 공유"""
    def __init__(self):
        self._lock    = threading.Lock()
        self._x       = -1
        self._y       = -1
        self._visible = False

    def update(self, x: int, y: int):
        with self._lock:
            self._x       = x
            self._y       = y
            self._visible = True

    def hide(self):
        with self._lock:
            self._visible = False

    def get(self):
        with self._lock:
            return self._x, self._y, self._visible


gaze_cursor_state = GazeCursorState()


def _make_gaze_cursor_view_class(screen_h: int):
    try:
        import objc
        from AppKit import NSView

        class GazeCursorViewImpl(NSView):

            def initWithFrame_(self, frame):
                self = objc.super(GazeCursorViewImpl, self).initWithFrame_(frame)
                if self is None:
                    return None
                self._sh = screen_h
                return self

            def drawRect_(self, rect):
                x, y, visible = gaze_cursor_state.get()
                if not visible or x < 0:
                    return
                try:
                    self._draw_cursor(x, y)
                except Exception:
                    pass

            def _draw_cursor(self, gx: int, gy: int):
                from AppKit import NSColor, NSBezierPath
                from Foundation import NSMakeRect

                # macOS 좌표계: 화면 좌상단 → NSWindow 좌하단 원점 변환
                ny = self._sh - gy

                # 외곽 링 (시안)
                r_outer = 22
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.0, 0.85, 1.0, 0.55
                ).set()
                outer = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(gx - r_outer, ny - r_outer, r_outer * 2, r_outer * 2)
                )
                outer.setLineWidth_(2.0)
                outer.stroke()

                # 내부 채운 원
                r_inner = 7
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.0, 0.85, 1.0, 0.85
                ).set()
                inner = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(gx - r_inner, ny - r_inner, r_inner * 2, r_inner * 2)
                )
                inner.fill()

                # 십자선
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.55
                ).set()
                cross = NSBezierPath.bezierPath()
                cross.moveToPoint_((gx - 14, ny))
                cross.lineToPoint_((gx + 14, ny))
                cross.moveToPoint_((gx, ny - 14))
                cross.lineToPoint_((gx, ny + 14))
                cross.setLineWidth_(1.0)
                cross.stroke()

        return GazeCursorViewImpl

    except Exception as e:
        print(f"[GazeCursorOverlay] View 생성 실패: {e}")
        return None


class GazeCursorOverlay:
    """실제 화면 위에 시선 커서를 그리는 투명 NSWindow 오버레이"""

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

            screen        = NSScreen.mainScreen()
            screen_frame  = screen.frame()
            sh            = int(screen_frame.size.height)

            ViewClass = _make_gaze_cursor_view_class(sh)
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
            self._window.setLevel_(997)          # 테두리(999)/호버(998) 아래, 일반 창 위
            self._window.setIgnoresMouseEvents_(True)
            self._window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )

            self._view = ViewClass.alloc().initWithFrame_(screen_frame)
            self._window.setContentView_(self._view)
            self._window.makeKeyAndOrderFront_(None)
            self._active = True
            print("[GazeCursorOverlay] 시선 커서 오버레이 시작")

        except ImportError:
            print("[GazeCursorOverlay] PyObjC 미설치 — 오버레이 건너뜀")
        except Exception as e:
            print(f"[GazeCursorOverlay] 시작 실패: {e}")

    def update(self, x: int, y: int):
        """시선 좌표 업데이트 + 화면 갱신"""
        gaze_cursor_state.update(x, y)
        if self._view and self._active:
            try:
                self._view.setNeedsDisplay_(True)
            except Exception:
                pass

    def hide_cursor(self):
        """커서 숨기기"""
        gaze_cursor_state.hide()
        if self._view and self._active:
            try:
                self._view.setNeedsDisplay_(True)
            except Exception:
                pass

    def stop(self):
        gaze_cursor_state.hide()
        if self._window and self._active:
            try:
                self._window.orderOut_(None)
            except Exception:
                pass
        self._active = False
        print("[GazeCursorOverlay] 시선 커서 오버레이 종료")
