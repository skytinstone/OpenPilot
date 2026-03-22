"""
호버 오버레이 모듈
macOS 화면 위에 호버 효과를 실시간으로 그림

효과 구성:
  1. 펄싱 링 (pulsing ring)     — 요소 테두리에 파란색 링이 숨쉬듯 확장
  2. 반투명 하이라이트           — 요소 영역을 부드럽게 채움
  3. 드웰 프로그레스 링          — 얼마나 오래 보고 있는지 원형 진행바
  4. 요소 레이블 툴팁            — 요소 이름을 말풍선 형태로 표시

중요: start() 는 반드시 메인 스레드에서 호출해야 합니다.
     refresh() 를 메인 루프에서 매 프레임 호출해 화면을 갱신합니다.
"""
import threading
import time
import math
from typing import Optional, Tuple

from core.vision.hover_detector import HoverElement


class HoverState:
    """스레드 간 호버 상태 공유 (lock 보호)"""
    def __init__(self):
        self._lock = threading.Lock()
        self._element: Optional[HoverElement] = None
        self._anim_start = time.time()

    def set(self, element: Optional[HoverElement]):
        with self._lock:
            if element is None:
                self._element = None
            elif self._element is None or not self._element.is_same_element(element):
                self._element = element
                self._anim_start = time.time()
            else:
                self._element = element   # dwell 업데이트

    def get(self) -> Tuple[Optional[HoverElement], float]:
        """(element, anim_elapsed) 반환"""
        with self._lock:
            return self._element, time.time() - self._anim_start


# 공유 상태 싱글톤
hover_state = HoverState()


class HoverOverlay:
    """
    macOS 화면 위에 호버 효과를 그리는 투명 오버레이 창

    주의: start() 는 메인 스레드에서 호출해야 합니다.
    매 프레임 refresh() 를 메인 루프에서 호출하면 화면이 갱신됩니다.
    """

    def __init__(self, dwell_threshold: float = 0.8):
        self._dwell_threshold = dwell_threshold
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

            HoverViewClass = _make_hover_view_class(sw, sh, self._dwell_threshold)
            if HoverViewClass is None:
                return

            self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                screen_frame,
                NSBorderlessWindowMask,
                NSBackingStoreBuffered,
                False,
            )
            self._window.setOpaque_(False)
            self._window.setBackgroundColor_(NSColor.clearColor())
            self._window.setLevel_(998)              # 빨간 테두리(999) 바로 아래
            self._window.setIgnoresMouseEvents_(True)
            self._window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )

            self._view = HoverViewClass.alloc().initWithFrame_(screen_frame)
            self._window.setContentView_(self._view)
            self._window.makeKeyAndOrderFront_(None)

            self._active = True
            print("[HoverOverlay] 호버 오버레이 시작")

        except ImportError:
            print("[HoverOverlay] PyObjC 미설치. 호버 오버레이를 건너뜁니다.")
        except Exception as e:
            print(f"[HoverOverlay] 오류: {e}")

    def refresh(self):
        """메인 루프에서 매 프레임 호출 — 오버레이 화면 갱신 요청"""
        if self._view is not None and self._active:
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
        print("[HoverOverlay] 호버 오버레이 종료")


def _make_hover_view_class(screen_w: int, screen_h: int, dwell_threshold: float):
    """호버 효과를 그리는 NSView 서브클래스 동적 생성"""
    try:
        import objc
        from AppKit import NSView, NSColor, NSBezierPath, NSFont, NSString

        class HoverViewImpl(NSView):

            def initWithFrame_(self, frame):
                self = objc.super(HoverViewImpl, self).initWithFrame_(frame)
                if self is None:
                    return None
                self._sw = screen_w
                self._sh = screen_h
                self._dwell_threshold = dwell_threshold
                return self

            def isFlipped(self):
                # macOS 좌표계: 좌하단 원점 → isFlipped=True 로 좌상단 원점으로 변환
                return True

            def drawRect_(self, rect):
                element, anim_elapsed = hover_state.get()
                if element is None:
                    return
                try:
                    self._draw_hover(element, anim_elapsed)
                except Exception:
                    pass

            def _draw_hover(self, element: HoverElement, anim_elapsed: float):
                from AppKit import NSColor, NSBezierPath, NSFont, NSString, NSAttributedString
                from Foundation import NSMakeRect, NSDictionary

                x, y, w, h = element.frame
                dwell = element.dwell_time
                progress = min(dwell / self._dwell_threshold, 1.0)

                # ─── 1. 반투명 하이라이트 배경 ───────────────────────
                alpha = 0.12 + 0.08 * math.sin(anim_elapsed * 3)
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.3, 0.6, 1.0, alpha
                ).set()
                bg_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, y, w, h), 8.0, 8.0
                )
                bg_path.fill()

                # ─── 2. 테두리 링 (펄싱) ────────────────────────────
                pulse = 1.0 + 0.06 * math.sin(anim_elapsed * 4)
                margin = 4 * pulse
                ring_rect = NSMakeRect(
                    x - margin, y - margin,
                    w + margin * 2, h + margin * 2
                )
                ring_alpha = 0.7 + 0.3 * math.sin(anim_elapsed * 4)
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.4, 0.75, 1.0, ring_alpha
                ).set()
                ring_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ring_rect, 10.0, 10.0
                )
                ring_path.setLineWidth_(2.0)
                ring_path.stroke()

                # ─── 3. 드웰 프로그레스 링 ──────────────────────────
                if progress < 1.0:
                    self._draw_dwell_ring(x, y, w, h, progress)
                else:
                    flash_alpha = max(0.0, 0.6 - (dwell - self._dwell_threshold) * 2)
                    if flash_alpha > 0:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            1.0, 1.0, 1.0, flash_alpha
                        ).set()
                        flash_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                            NSMakeRect(x - 6, y - 6, w + 12, h + 12), 12.0, 12.0
                        )
                        flash_path.setLineWidth_(3.0)
                        flash_path.stroke()

                # ─── 4. 레이블 툴팁 ─────────────────────────────────
                if element.title:
                    self._draw_tooltip(element.title, x, y, w)

            def _draw_dwell_ring(self, x, y, w, h, progress):
                from AppKit import NSColor, NSBezierPath
                from Foundation import NSMakeRect
                import math

                cx = x + w + 12
                cy = y - 12
                radius = 10.0

                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.2, 0.2, 0.6).set()
                bg = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2)
                )
                bg.fill()

                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.4, 0.85, 1.0, 0.95).set()
                arc = NSBezierPath.bezierPath()
                start_angle = 90.0
                end_angle = 90.0 - (360.0 * progress)
                arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                    (cx, cy), radius - 2, start_angle, end_angle, True
                )
                arc.setLineWidth_(2.5)
                arc.setLineCapStyle_(1)
                arc.stroke()

            def _draw_tooltip(self, title: str, x, y, w):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString, NSForegroundColorAttributeName,
                    NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                font = NSFont.systemFontOfSize_(11.0)
                attrs = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName: NSColor.whiteColor(),
                }
                ns_title = NSString.stringWithString_(title[:30])
                attr_str = NSAttributedString.alloc().initWithString_attributes_(ns_title, attrs)
                text_size = attr_str.size()

                pad = 6.0
                tip_w = text_size.width + pad * 2
                tip_h = text_size.height + pad * 2
                tip_x = x + (w - tip_w) / 2
                tip_y = y - tip_h - 8

                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.82).set()
                bubble = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(tip_x, tip_y, tip_w, tip_h), 5.0, 5.0
                )
                bubble.fill()

                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.5, 0.5, 0.5, 0.5).set()
                bubble.setLineWidth_(0.8)
                bubble.stroke()

                attr_str.drawAtPoint_(NSMakePoint(tip_x + pad, tip_y + pad))

        return HoverViewImpl

    except Exception as e:
        print(f"[HoverOverlay] HoverView 클래스 생성 실패: {e}")
        return None
