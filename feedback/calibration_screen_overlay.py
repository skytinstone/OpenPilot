"""
실제 화면 전체에 캘리브레이션 포인트를 표시하는 NSWindow 오버레이

- 반투명 어두운 전체화면 배경
- 목표 포인트 + 진행률 호 + 안내 텍스트
- 캘리브레이션 완료 후 자동 숨김

사용법:
    overlay = CalibrationScreenOverlay()
    overlay.start()                         # 메인 스레드에서 호출
    overlay.show_point(tx, ty, prog, i, n)  # 매 프레임 업데이트
    overlay.hide()                          # 캘리브레이션 완료 후
    overlay.stop()
"""
import threading
from dataclasses import dataclass


@dataclass
class _CalData:
    active:      bool  = False
    target_x:    float = 0.5    # 정규화 0~1
    target_y:    float = 0.5
    progress:    float = 0.0
    current_idx: int   = 0
    total:       int   = 5


class _CalibrationScreenState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = _CalData()

    def show(self, tx: float, ty: float, progress: float, idx: int, total: int):
        with self._lock:
            self._data = _CalData(
                active=True, target_x=tx, target_y=ty,
                progress=progress, current_idx=idx, total=total,
            )

    def hide(self):
        with self._lock:
            self._data = _CalData(active=False)

    def get(self) -> _CalData:
        with self._lock:
            d = self._data
            return _CalData(
                active=d.active, target_x=d.target_x, target_y=d.target_y,
                progress=d.progress, current_idx=d.current_idx, total=d.total,
            )


_cal_state = _CalibrationScreenState()


def _make_cal_view_class(screen_w: int, screen_h: int):
    try:
        import objc
        from AppKit import NSView

        class CalViewImpl(NSView):

            def initWithFrame_(self, frame):
                self = objc.super(CalViewImpl, self).initWithFrame_(frame)
                if self is None:
                    return None
                self._sw = screen_w
                self._sh = screen_h
                return self

            def drawRect_(self, rect):
                data = _cal_state.get()
                if not data.active:
                    return
                try:
                    self._draw(data)
                except Exception:
                    pass

            def _draw(self, data: _CalData):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                sw, sh = self._sw, self._sh

                # ── 반투명 어두운 배경 ──────────────────────────────
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.0, 0.0, 0.0, 0.72
                ).set()
                NSBezierPath.fillRect_(NSMakeRect(0, 0, sw, sh))

                # 목표 포인트 좌표 (macOS y 뒤집기: 좌상단→좌하단)
                tx = data.target_x * sw
                ty = sh - (data.target_y * sh)

                # ── 외곽 원 배경 ────────────────────────────────────
                r = 34
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.25, 0.25, 0.25, 0.85
                ).set()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(tx - r, ty - r, r * 2, r * 2)
                ).fill()

                # ── 흰색 테두리 링 ──────────────────────────────────
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.9
                ).set()
                ring = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(tx - r, ty - r, r * 2, r * 2)
                )
                ring.setLineWidth_(2.0)
                ring.stroke()

                # ── 진행률 초록 호 ──────────────────────────────────
                if data.progress > 0.001:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        0.15, 1.0, 0.45, 0.95
                    ).set()
                    arc = NSBezierPath.bezierPath()
                    arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                        (tx, ty), r - 5,
                        90.0, 90.0 - 360.0 * data.progress,
                        True,
                    )
                    arc.setLineWidth_(6.0)
                    arc.setLineCapStyle_(1)    # NSRoundLineCapStyle
                    arc.stroke()

                # ── 중앙 흰 점 ──────────────────────────────────────
                NSColor.whiteColor().set()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(tx - 5, ty - 5, 10, 10)
                ).fill()

                # ── 안내 텍스트 (하단 중앙) ──────────────────────────
                text  = f"Look at this point  ({data.current_idx + 1} / {data.total})"
                font  = NSFont.boldSystemFontOfSize_(20.0)
                attrs = {
                    NSFontAttributeName:        font,
                    NSForegroundColorAttributeName: NSColor.whiteColor(),
                }
                ns_str   = NSString.stringWithString_(text)
                attr_str = NSAttributedString.alloc().initWithString_attributes_(
                    ns_str, attrs
                )
                ts = attr_str.size()
                attr_str.drawAtPoint_(NSMakePoint((sw - ts.width) / 2, 50))

        return CalViewImpl

    except Exception as e:
        print(f"[CalibrationScreenOverlay] View 생성 실패: {e}")
        return None


class CalibrationScreenOverlay:
    """실제 화면 전체에 캘리브레이션 포인트를 표시하는 NSWindow 오버레이"""

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
            sw            = int(screen_frame.size.width)
            sh            = int(screen_frame.size.height)

            ViewClass = _make_cal_view_class(sw, sh)
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
            self._window.setLevel_(1000)          # 최상위 — 모든 창 위
            self._window.setIgnoresMouseEvents_(True)
            self._window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )

            self._view = ViewClass.alloc().initWithFrame_(screen_frame)
            self._window.setContentView_(self._view)
            self._window.orderOut_(None)           # 처음엔 숨김
            self._active = True
            print("[CalibrationScreenOverlay] 준비 완료")

        except ImportError:
            print("[CalibrationScreenOverlay] PyObjC 미설치 — 오버레이 건너뜀")
        except Exception as e:
            print(f"[CalibrationScreenOverlay] 시작 실패: {e}")

    def show_point(self, target_x: float, target_y: float,
                   progress: float, idx: int, total: int):
        """캘리브레이션 포인트 표시 및 갱신 (매 프레임 호출)"""
        _cal_state.show(target_x, target_y, progress, idx, total)
        if self._window and self._active:
            try:
                self._window.makeKeyAndOrderFront_(None)
                self._view.setNeedsDisplay_(True)
            except Exception:
                pass

    def hide(self):
        """캘리브레이션 완료 후 오버레이 숨기기"""
        _cal_state.hide()
        if self._window and self._active:
            try:
                self._window.orderOut_(None)
            except Exception:
                pass

    def refresh(self):
        if self._view and self._active:
            try:
                self._view.setNeedsDisplay_(True)
            except Exception:
                pass

    def stop(self):
        _cal_state.hide()
        if self._window and self._active:
            try:
                self._window.orderOut_(None)
            except Exception:
                pass
        self._active = False
        print("[CalibrationScreenOverlay] 종료")
