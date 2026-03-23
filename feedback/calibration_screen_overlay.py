"""
실제 화면 전체에 캘리브레이션 UI를 표시하는 NSWindow 오버레이 (Enhanced v2)

단계별 UI:
  prepare  — 반투명 배경 + 안내 텍스트 (2초)
  aiming   — 포인트 표시 + "시선을 맞추세요" (흰색 링)
  sampling — 안정 표시등(녹색) + 샘플 진행 바
  done_pt  — 포인트 완료 초록 플래시
  validate — 검증 포인트 (파란색)
  result   — 정확도 결과 표시
"""
import threading
import time
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class _CalData:
    active:           bool  = False
    phase:            str   = "prepare"
    target_x:         float = 0.5
    target_y:         float = 0.5
    progress:         float = 0.0       # 0~1 (안정 샘플 진행률)
    current_idx:      int   = 0
    total:            int   = 9
    is_stable:        bool  = False
    stable_count:     int   = 0
    stable_required:  int   = 30
    point_qualities:  List[float] = field(default_factory=list)
    accuracy_px:      float = -1.0
    is_validation:    bool  = False


class _CalibrationState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = _CalData()

    def update(self, data: _CalData):
        with self._lock:
            self._data = data

    def hide(self):
        with self._lock:
            self._data = _CalData(active=False)

    def get(self) -> _CalData:
        with self._lock:
            d = self._data
            return _CalData(
                active=d.active, phase=d.phase,
                target_x=d.target_x, target_y=d.target_y,
                progress=d.progress, current_idx=d.current_idx,
                total=d.total, is_stable=d.is_stable,
                stable_count=d.stable_count,
                stable_required=d.stable_required,
                point_qualities=list(d.point_qualities),
                accuracy_px=d.accuracy_px,
                is_validation=d.is_validation,
            )


_cal_state = _CalibrationState()


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
                now    = time.time()

                # ── 반투명 배경 ─────────────────────────────────────
                bg_alpha = 0.78
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.0, 0.0, 0.0, bg_alpha
                ).set()
                NSBezierPath.fillRect_(NSMakeRect(0, 0, sw, sh))

                phase = data.phase

                # ── 준비 단계 ────────────────────────────────────────
                if phase == "prepare":
                    self._draw_prepare(data, sw, sh)
                    return

                # ── 결과 단계 ────────────────────────────────────────
                if phase == "result":
                    self._draw_result(data, sw, sh)
                    return

                # ── 포인트 표시 (aiming / sampling / done_pt / validate) ──
                tx = data.target_x * sw
                ty = sh - (data.target_y * sh)   # macOS: y=0 at bottom

                is_val    = data.is_validation
                is_done   = (phase == "done_pt")
                is_stable = data.is_stable or is_done

                # 외곽 원 크기 (done_pt 때 작아짐)
                r_outer = 34 if not is_done else 20

                # 색상 선택
                if is_val:
                    ring_r, ring_g, ring_b = 0.2, 0.6, 1.0   # 파랑 (검증)
                elif is_done:
                    ring_r, ring_g, ring_b = 0.15, 1.0, 0.45  # 초록 (완료)
                elif is_stable:
                    ring_r, ring_g, ring_b = 0.15, 1.0, 0.45  # 초록 (안정)
                else:
                    ring_r, ring_g, ring_b = 1.0, 1.0, 1.0    # 흰색 (대기)

                # ── 외곽 원 배경 ────────────────────────────────────
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.18, 0.18, 0.18, 0.9
                ).set()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(tx - r_outer, ty - r_outer,
                               r_outer * 2, r_outer * 2)
                ).fill()

                # ── 링 ──────────────────────────────────────────────
                ring_alpha = 0.95
                if is_stable and not is_done:
                    # 안정 시 펄스
                    pulse = (math.sin(now * 6) + 1) / 2
                    ring_alpha = 0.7 + 0.3 * pulse

                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    ring_r, ring_g, ring_b, ring_alpha
                ).set()
                ring_path = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(tx - r_outer, ty - r_outer,
                               r_outer * 2, r_outer * 2)
                )
                ring_path.setLineWidth_(2.5)
                ring_path.stroke()

                # ── 진행 호 (샘플 진행률) ──────────────────────────
                if data.progress > 0.001 and not is_val:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        ring_r, ring_g, ring_b, 0.95
                    ).set()
                    arc = NSBezierPath.bezierPath()
                    end_angle = 90.0 - 360.0 * data.progress
                    arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                        (tx, ty), r_outer - 5,
                        90.0, end_angle, True,
                    )
                    arc.setLineWidth_(7.0)
                    arc.setLineCapStyle_(1)
                    arc.stroke()

                # ── 중앙 점 ─────────────────────────────────────────
                dot_r = 5 if not is_done else 3
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    ring_r, ring_g, ring_b, 1.0
                ).set()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(tx - dot_r, ty - dot_r,
                               dot_r * 2, dot_r * 2)
                ).fill()

                # ── 안정 표시등 (포인트 위) ──────────────────────────
                if not is_val and not is_done:
                    self._draw_stability_badge(
                        tx, ty + r_outer + 18,
                        is_stable, data.stable_count, data.stable_required,
                    )

                # ── 하단 안내 텍스트 ────────────────────────────────
                if is_val:
                    msg = (f"Validation  {data.current_idx - data.total + 1}"
                           f" / {len([1,2,3,4])}")
                elif is_done:
                    q = data.point_qualities[-1] if data.point_qualities else 0
                    stars = "★★★" if q > 0.7 else "★★" if q > 0.4 else "★"
                    msg = f"Point {data.current_idx + 1} complete  {stars}"
                elif is_stable:
                    msg = (f"Collecting...  {data.stable_count} / "
                           f"{data.stable_required}")
                else:
                    msg = (f"Point {data.current_idx + 1} / {data.total}"
                           f"  —  Gaze at the dot and hold still")
                self._draw_bottom_text(msg, sw, sh)

                # ── 포인트 완료 품질 히스토리 바 ────────────────────
                if data.point_qualities:
                    self._draw_quality_history(data.point_qualities, sw, sh)

            # ── 준비 화면 ────────────────────────────────────────────

            def _draw_prepare(self, data, sw, sh):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                lines = [
                    ("Eye Calibration", 28.0, (1.0, 1.0, 1.0)),
                    ("", 12.0, (0.7, 0.7, 0.7)),
                    (f"{'9' if data.total == 9 else '5'}-Point Calibration", 18.0, (0.8, 0.9, 1.0)),
                    ("Look at each dot and hold still until it fills green.", 15.0, (0.7, 0.7, 0.7)),
                    ("Keep your head still — only move your eyes.", 15.0, (0.7, 0.7, 0.7)),
                    ("Starting in a moment...", 14.0, (0.5, 0.9, 0.5)),
                ]
                y = sh / 2 + len(lines) * 16 + 10
                for text, size, color in lines:
                    if not text:
                        y -= 10
                        continue
                    font  = NSFont.boldSystemFontOfSize_(size) if size >= 18 \
                            else NSFont.systemFontOfSize_(size)
                    ns    = NSString.stringWithString_(text)
                    attrs = {
                        NSFontAttributeName: font,
                        NSForegroundColorAttributeName:
                            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                                color[0], color[1], color[2], 0.95),
                    }
                    astr = NSAttributedString.alloc().initWithString_attributes_(ns, attrs)
                    ts   = astr.size()
                    astr.drawAtPoint_(NSMakePoint((sw - ts.width) / 2, y))
                    y -= (size + 10)

            # ── 결과 화면 ────────────────────────────────────────────

            def _draw_result(self, data, sw, sh):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                acc = data.accuracy_px
                if acc > 0:
                    acc_str = f"{acc:.0f} px"
                    if acc < 50:
                        acc_color = (0.2, 1.0, 0.4)
                        grade = "Excellent"
                    elif acc < 100:
                        acc_color = (1.0, 0.85, 0.2)
                        grade = "Good"
                    else:
                        acc_color = (1.0, 0.4, 0.3)
                        grade = "Fair — Try again (c key)"
                else:
                    acc_str, acc_color, grade = "--", (0.7, 0.7, 0.7), ""

                avg_q = (sum(data.point_qualities) / len(data.point_qualities)
                         if data.point_qualities else 0.0)

                lines = [
                    ("Calibration Complete!", 26.0, (1.0, 1.0, 1.0)),
                    ("", 0, None),
                    (f"Accuracy : {acc_str}  {grade}", 18.0, acc_color),
                    (f"Avg Quality : {avg_q:.0%}", 16.0, (0.7, 0.8, 1.0)),
                    ("", 0, None),
                    ("Calibration saved automatically.", 13.0, (0.5, 0.5, 0.5)),
                    ("Press  c  to recalibrate.", 13.0, (0.5, 0.5, 0.5)),
                ]
                y = sh / 2 + 120
                for text, size, color in lines:
                    if not text:
                        y -= 14
                        continue
                    bold = size >= 18
                    font = NSFont.boldSystemFontOfSize_(size) if bold \
                           else NSFont.systemFontOfSize_(size)
                    ns   = NSString.stringWithString_(text)
                    attrs = {
                        NSFontAttributeName: font,
                        NSForegroundColorAttributeName:
                            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                                color[0], color[1], color[2], 0.95),
                    }
                    astr = NSAttributedString.alloc().initWithString_attributes_(ns, attrs)
                    ts   = astr.size()
                    astr.drawAtPoint_(NSMakePoint((sw - ts.width) / 2, y))
                    y -= (size + 12)

            # ── 안정 배지 ────────────────────────────────────────────

            def _draw_stability_badge(self, cx, cy,
                                       is_stable, count, required):
                from AppKit import (
                    NSColor, NSBezierPath, NSFont, NSString,
                    NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakeRect, NSMakePoint

                label = f"STABLE  {count}/{required}" if is_stable \
                        else "Hold still..."
                r, g, b = (0.15, 1.0, 0.45) if is_stable else (0.8, 0.8, 0.8)
                alpha   = 0.95

                font  = NSFont.boldSystemFontOfSize_(12.0)
                ns    = NSString.stringWithString_(label)
                attrs = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            r, g, b, alpha),
                }
                astr = NSAttributedString.alloc().initWithString_attributes_(ns, attrs)
                ts   = astr.size()
                astr.drawAtPoint_(NSMakePoint(cx - ts.width / 2, cy))

            # ── 하단 안내 텍스트 ─────────────────────────────────────

            def _draw_bottom_text(self, text, sw, sh):
                from AppKit import (
                    NSColor, NSFont, NSString, NSAttributedString,
                    NSForegroundColorAttributeName, NSFontAttributeName,
                )
                from Foundation import NSMakePoint

                font  = NSFont.systemFontOfSize_(16.0)
                ns    = NSString.stringWithString_(text)
                attrs = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName:
                        NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            0.95, 0.95, 0.95, 0.9),
                }
                astr = NSAttributedString.alloc().initWithString_attributes_(ns, attrs)
                ts   = astr.size()
                astr.drawAtPoint_(NSMakePoint((sw - ts.width) / 2, 50.0))

            # ── 포인트 품질 히스토리 (하단 바) ───────────────────────

            def _draw_quality_history(self, qualities, sw, sh):
                from AppKit import NSColor, NSBezierPath
                from Foundation import NSMakeRect

                n    = len(qualities)
                bw   = 24
                gap  = 6
                total_w = n * (bw + gap)
                start_x = (sw - total_w) / 2
                by      = 24.0

                for i, q in enumerate(qualities):
                    bx = start_x + i * (bw + gap)
                    if q > 0.7:
                        r, g, b = 0.15, 1.0, 0.45
                    elif q > 0.4:
                        r, g, b = 1.0, 0.85, 0.2
                    else:
                        r, g, b = 1.0, 0.3, 0.3

                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        r, g, b, 0.85
                    ).set()
                    filled_h = max(4, int(18 * q))
                    NSBezierPath.fillRect_(
                        NSMakeRect(bx, by, bw, filled_h)
                    )
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        r, g, b, 0.4
                    ).set()
                    bg = NSBezierPath.bezierPathWithRect_(NSMakeRect(bx, by, bw, 18))
                    bg.setLineWidth_(1.0)
                    bg.stroke()

        return CalViewImpl

    except Exception as e:
        print(f"[CalibrationScreenOverlay] View 생성 실패: {e}")
        return None


class CalibrationScreenOverlay:
    """실제 화면 전체 캘리브레이션 NSWindow 오버레이 (Enhanced v2)"""

    def __init__(self):
        self._window = None
        self._view   = None
        self._active = False

    def start(self):
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

            ViewClass = _make_cal_view_class(sw, sh)
            if ViewClass is None:
                return

            self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                screen_frame, NSBorderlessWindowMask, NSBackingStoreBuffered, False,
            )
            self._window.setOpaque_(False)
            self._window.setBackgroundColor_(NSColor.clearColor())
            self._window.setLevel_(1000)
            self._window.setIgnoresMouseEvents_(True)
            self._window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )
            self._view = ViewClass.alloc().initWithFrame_(screen_frame)
            self._window.setContentView_(self._view)
            self._window.orderOut_(None)
            self._active = True
            print("[CalibrationScreenOverlay] 준비 완료 (Enhanced v2)")

        except ImportError:
            print("[CalibrationScreenOverlay] PyObjC 미설치 — 건너뜀")
        except Exception as e:
            print(f"[CalibrationScreenOverlay] 시작 실패: {e}")

    def update_from_status(self, status: dict):
        """GazeEstimator.update_calibration() 반환값으로 오버레이 업데이트"""
        phase = status.get("phase", "aiming")
        total = status.get("total", 9)
        idx   = status.get("current_idx", 0)

        # 검증 단계인지 판단
        is_val = (phase == "validate")

        if is_val:
            from core.vision.gaze_estimator import VALIDATION_POSITIONS
            vi = status.get("current_idx", 0) - total
            vi = max(0, min(vi, len(VALIDATION_POSITIONS) - 1))
            vp = VALIDATION_POSITIONS[vi]
            tx, ty = vp
        else:
            # 현재 포인트 좌표 가져오기 (caller가 넘겨줌 or default)
            tx = status.get("target_x", 0.5)
            ty = status.get("target_y", 0.5)

        data = _CalData(
            active          = True,
            phase           = phase,
            target_x        = tx,
            target_y        = ty,
            progress        = status.get("progress", 0.0),
            current_idx     = idx,
            total           = total,
            is_stable       = status.get("is_stable", False),
            stable_count    = status.get("stable_count", 0),
            stable_required = status.get("stable_required", 30),
            point_qualities = status.get("point_qualities", []),
            accuracy_px     = status.get("accuracy_px", -1.0),
            is_validation   = is_val,
        )
        _cal_state.update(data)
        if self._window and self._active:
            try:
                self._window.makeKeyAndOrderFront_(None)
                self._view.setNeedsDisplay_(True)
            except Exception:
                pass

    def show_point(self, target_x: float, target_y: float,
                   progress: float, idx: int, total: int):
        """하위 호환 인터페이스"""
        data = _CalData(
            active=True, phase="aiming",
            target_x=target_x, target_y=target_y,
            progress=progress, current_idx=idx, total=total,
        )
        _cal_state.update(data)
        if self._window and self._active:
            try:
                self._window.makeKeyAndOrderFront_(None)
                self._view.setNeedsDisplay_(True)
            except Exception:
                pass

    def hide(self):
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
