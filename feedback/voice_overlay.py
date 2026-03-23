"""
실제 화면 위에 음성 인식 상태를 표시하는 NSWindow 오버레이

표시 항목:
  - 우측 상단: 모드 배지 (DICTATE=초록 / AI CMD=파랑), REC 표시등 + 마이크 레벨 바
  - 화면 하단 중앙: 인식된 텍스트 토스트 (3초 후 fade-out)
  - 화면 하단: 상태 메시지 (Transcribing... / Done. 등)

사용법:
    overlay = VoiceOverlay()
    overlay.start()
    overlay.update(recording=True, mode="DICTATE", transcription="", status="Recording...", mic_level=0.5)
    overlay.refresh()   # 매 프레임
    overlay.stop()
"""
import threading
import time
import math


# ── 공유 상태 ─────────────────────────────────────────────────────

class _VoiceState:
    def __init__(self):
        self._lock         = threading.Lock()
        self.recording     = False
        self.mode          = "DICTATE"
        self.transcription = ""
        self.status        = ""
        self.mic_level     = 0.0
        self._trans_expire = 0.0    # 텍스트 표시 만료 시간

    def update(self, recording: bool, mode: str,
               transcription: str, status: str, mic_level: float,
               trans_duration: float = 3.0):
        with self._lock:
            self.recording  = recording
            self.mode       = mode
            self.status     = status
            self.mic_level  = mic_level
            if transcription != self.transcription:
                self.transcription  = transcription
                if transcription:
                    self._trans_expire = time.time() + trans_duration
            elif not transcription:
                self._trans_expire = 0.0

    def get(self):
        with self._lock:
            remaining = max(0.0, self._trans_expire - time.time())
            return (
                self.recording,
                self.mode,
                self.transcription if remaining > 0 else "",
                self.status,
                self.mic_level,
                remaining,
            )


_voice_state = _VoiceState()


# ── NSView 구현 ───────────────────────────────────────────────────

def _make_voice_view_class(screen_w: int, screen_h: int):
    try:
        import objc
        from AppKit import NSView

        class VoiceViewImpl(NSView):

            def initWithFrame_(self, frame):
                self = objc.super(VoiceViewImpl, self).initWithFrame_(frame)
                if self is None:
                    return None
                self._sw = screen_w
                self._sh = screen_h
                return self

            def drawRect_(self, rect):
                recording, mode, transcription, status, mic_level, trans_remaining = \
                    _voice_state.get()
                try:
                    self._draw(recording, mode, transcription,
                               status, mic_level, trans_remaining)
                except Exception:
                    pass

            def _draw(self, recording, mode, transcription,
                      status, mic_level, trans_remaining):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                sw, sh = self._sw, self._sh
                now = time.time()

                # ── 모드 배지 (우측 상단) ─────────────────────────
                mode_color = (0.25, 0.85, 0.35) if mode == "DICTATE" \
                             else (0.15, 0.65, 1.0)
                self._draw_badge(
                    f"● {mode}",
                    x=sw - 160, y=sh - 48,
                    color=mode_color, alpha=0.88,
                    font_size=14.0, pad=10.0,
                )

                # ── REC 표시등 (우측 상단, 모드 배지 아래) ───────
                if recording:
                    # 깜빡이는 빨간 원
                    pulse = (math.sin(now * 6) + 1) / 2   # 0~1
                    rec_alpha = 0.6 + 0.4 * pulse
                    self._draw_badge(
                        "⏺  REC",
                        x=sw - 130, y=sh - 88,
                        color=(1.0, 0.15, 0.15), alpha=rec_alpha,
                        font_size=14.0, pad=10.0,
                    )

                    # 마이크 레벨 바 (우측 세로)
                    self._draw_mic_bar(mic_level, sw, sh)

                # ── 상태 메시지 (화면 하단 중앙, 작은 회색 텍스트) ──
                if status and not transcription:
                    self._draw_status(status, sw, sh)

                # ── 인식된 텍스트 토스트 (화면 하단 중앙) ─────────
                if transcription:
                    fade_alpha = min(1.0, trans_remaining / 0.5)
                    self._draw_transcription(transcription, sw, sh,
                                             alpha=fade_alpha, mode=mode)

            def _draw_badge(self, text, x, y, color, alpha,
                            font_size=15.0, pad=12.0):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                r, g, b = color
                font   = NSFont.boldSystemFontOfSize_(font_size)
                ns_str = NSString.stringWithString_(text)
                attrs  = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            1.0, 1.0, 1.0, alpha),
                }
                astr = NSAttributedString.alloc() \
                    .initWithString_attributes_(ns_str, attrs)
                ts   = astr.size()

                bw = ts.width  + pad * 2
                bh = ts.height + pad * 1.2

                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.06, 0.06, 0.06, 0.78 * alpha
                ).set()
                bg = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, y, bw, bh), 8.0, 8.0
                )
                bg.fill()

                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    r, g, b, 0.85 * alpha
                ).set()
                bg.setLineWidth_(1.8)
                bg.stroke()

                astr.drawAtPoint_(
                    NSMakePoint(x + pad, y + (bh - ts.height) / 2)
                )

            def _draw_mic_bar(self, level, sw, sh):
                from AppKit import NSColor, NSBezierPath
                from Foundation import NSMakeRect

                bx    = sw - 18
                total = 120
                by    = sh - 96 - total
                fill  = int(total * min(level, 1.0))

                # 배경
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.1, 0.1, 0.1, 0.55
                ).set()
                NSBezierPath.fillRect_(NSMakeRect(bx - 8, by, 8, total))

                # 채워진 바
                if fill > 0:
                    r = 0.25 if level < 0.7 else 0.9
                    g = 0.75 if level < 0.7 else 0.2
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        r, g, 0.25, 0.85
                    ).set()
                    NSBezierPath.fillRect_(
                        NSMakeRect(bx - 8, by + total - fill, 8, fill)
                    )

            def _draw_status(self, status, sw, sh):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                font   = NSFont.systemFontOfSize_(13.0)
                ns_str = NSString.stringWithString_(status)
                attrs  = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            0.75, 0.75, 0.75, 0.85),
                }
                astr = NSAttributedString.alloc() \
                    .initWithString_attributes_(ns_str, attrs)
                ts   = astr.size()
                tx   = (sw - ts.width) / 2
                ty   = 36.0
                astr.drawAtPoint_(NSMakePoint(tx, ty))

            def _draw_transcription(self, text, sw, sh, alpha, mode):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                # 길면 자르기
                max_chars = 70
                display = text if len(text) <= max_chars \
                          else "..." + text[-(max_chars - 3):]
                display = f'"{display}"'

                color = (0.25, 0.90, 0.40) if mode == "DICTATE" \
                        else (0.15, 0.70, 1.0)

                font   = NSFont.systemFontOfSize_(18.0)
                ns_str = NSString.stringWithString_(display)
                attrs  = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            1.0, 1.0, 1.0, alpha),
                }
                astr = NSAttributedString.alloc() \
                    .initWithString_attributes_(ns_str, attrs)
                ts   = astr.size()

                pad = 20.0
                bw  = ts.width  + pad * 2
                bh  = ts.height + pad * 1.5
                bx  = (sw - bw) / 2
                by  = 80.0     # 화면 하단에서 80pt

                r, g, b = color
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.05, 0.05, 0.05, 0.85 * alpha
                ).set()
                bg = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(bx, by, bw, bh), 14.0, 14.0
                )
                bg.fill()

                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    r, g, b, 0.9 * alpha
                ).set()
                bg.setLineWidth_(2.0)
                bg.stroke()

                astr.drawAtPoint_(
                    NSMakePoint(bx + pad, by + (bh - ts.height) / 2)
                )

        return VoiceViewImpl

    except Exception as e:
        print(f"[VoiceOverlay] View 생성 실패: {e}")
        return None


# ── VoiceOverlay 클래스 ───────────────────────────────────────────

class VoiceOverlay:
    """실제 화면 위에 음성 인식 상태를 표시하는 투명 NSWindow 오버레이"""

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
            sw = int(screen_frame.size.width)
            sh = int(screen_frame.size.height)

            ViewClass = _make_voice_view_class(sw, sh)
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
            print("[VoiceOverlay] 음성 오버레이 시작")

        except ImportError:
            print("[VoiceOverlay] PyObjC 미설치 — 건너뜀")
        except Exception as e:
            print(f"[VoiceOverlay] 시작 실패: {e}")

    def update(self, recording: bool, mode: str, transcription: str,
               status: str, mic_level: float):
        """매 프레임 상태 업데이트"""
        _voice_state.update(recording, mode, transcription, status, mic_level)

    def refresh(self):
        """매 프레임 호출 — 화면 갱신"""
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
        print("[VoiceOverlay] 음성 오버레이 종료")
