"""
화면 오버레이 모듈
Open Pilot 실행 중 상태를 시각적으로 표시:
  - 화면 테두리: 빨간색 (실행 중 상태 표시)

macOS NSWindow를 이용해 투명 오버레이 창을 최상위로 띄움

중요: start() 는 반드시 메인 스레드에서 호출해야 합니다.
     NSApplication.sharedApplication() 도 호출자가 먼저 처리해야 합니다.
"""
from typing import Tuple, Optional


def _make_border_view_class():
    """PyObjC NSView 서브클래스 동적 생성 (메인 스레드에서 호출)"""
    try:
        import objc
        from AppKit import NSView

        class BorderViewImpl(NSView):
            def initWithFrame_(self, frame):
                self = objc.super(BorderViewImpl, self).initWithFrame_(frame)
                if self is None:
                    return None
                self._bw = 5
                self._color = (1.0, 0.15, 0.15)
                self._sw = 1440
                self._sh = 900
                return self

            def set_params(self, border_width, color, screen_w, screen_h):
                self._bw = border_width
                self._color = color
                self._sw = screen_w
                self._sh = screen_h

            def drawRect_(self, rect):
                try:
                    from AppKit import NSColor, NSBezierPath
                    r, g, b = self._color
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.9).set()
                    bw = self._bw
                    sw, sh = self._sw, self._sh
                    paths = [
                        NSBezierPath.bezierPathWithRect_(((0, sh - bw), (sw, bw))),
                        NSBezierPath.bezierPathWithRect_(((0, 0), (sw, bw))),
                        NSBezierPath.bezierPathWithRect_(((0, 0), (bw, sh))),
                        NSBezierPath.bezierPathWithRect_(((sw - bw, 0), (bw, sh))),
                    ]
                    for p in paths:
                        p.fill()
                except Exception:
                    pass

        return BorderViewImpl
    except Exception as e:
        print(f"[ScreenOverlay] BorderView 클래스 생성 실패: {e}")
        return None


class ScreenBorderOverlayV2:
    """
    화면 전체를 덮는 투명 오버레이 창 — 테두리 색상으로 모드 구분

    모드별 색상:
      - Test Mode (기본): 빨간색 (1.0, 0.15, 0.15)
      - Real Mode:        초록색 (0.15, 0.85, 0.3)

    주의: start() 는 메인 스레드에서 호출해야 합니다.
    NSApplication.sharedApplication() 은 호출 전에 이미 처리되어 있어야 합니다.
    """

    # 프리셋 색상
    COLOR_TEST = (1.0, 0.15, 0.15)   # 빨간색 — Test Mode
    COLOR_REAL = (0.15, 0.85, 0.3)   # 초록색 — Real Mode

    def __init__(self, border_width: int = 5,
                 color: tuple = None,
                 mode: str = "test"):
        """
        Args:
            border_width: 테두리 두께 (px)
            color:        직접 RGB 튜플 지정 (0.0~1.0)
            mode:         "test" (빨강) / "real" (초록)
        """
        self._border_width = border_width
        if color is not None:
            self._border_color = color
        elif mode == "real":
            self._border_color = self.COLOR_REAL
        else:
            self._border_color = self.COLOR_TEST
        self._mode = mode
        self._window = None
        self._view = None
        self._active = False

    def start(self):
        """메인 스레드에서 NSWindow 직접 생성 (별도 스레드 없음)"""
        try:
            from AppKit import (
                NSWindow, NSColor,
                NSBorderlessWindowMask, NSBackingStoreBuffered,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorStationary,
                NSScreen,
            )

            screen = NSScreen.mainScreen()
            screen_frame = screen.frame()
            sw = int(screen_frame.size.width)
            sh = int(screen_frame.size.height)

            BorderViewClass = _make_border_view_class()
            if BorderViewClass is None:
                return

            self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                screen_frame,
                NSBorderlessWindowMask,
                NSBackingStoreBuffered,
                False,
            )
            self._window.setOpaque_(False)
            self._window.setBackgroundColor_(NSColor.clearColor())
            self._window.setLevel_(999)
            self._window.setIgnoresMouseEvents_(True)
            self._window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )

            view = BorderViewClass.alloc().initWithFrame_(screen_frame)
            view.set_params(self._border_width, self._border_color, sw, sh)
            self._window.setContentView_(view)
            self._view = view
            self._window.makeKeyAndOrderFront_(None)

            self._active = True
            label = "초록" if self._mode == "real" else "빨간"
            print(f"[ScreenOverlay] {label} 테두리 오버레이 시작 ({self._mode} mode)")

        except ImportError:
            print("[ScreenOverlay] PyObjC 미설치. 테두리 오버레이를 건너뜁니다.")
            print("              설치: pip install pyobjc-framework-Cocoa")
        except Exception as e:
            print(f"[ScreenOverlay] 오버레이 실행 오류: {e}")

    def stop(self):
        if self._window and self._active:
            try:
                self._window.orderOut_(None)
            except Exception:
                pass
        self._active = False
        print("[ScreenOverlay] 테두리 오버레이 종료")
