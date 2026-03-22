"""
Step 2 — Real Screen Hand Gesture Control
==========================================
손 제스처로 실제 화면의 클릭·스크롤·줌을 제어합니다.

카메라 테스트 창 대신 실제 데스크톱 화면에서 동작:
  - 제스처 이벤트(클릭/스크롤/줌)가 실제 화면 하단에 토스트로 표시됩니다
  - 화면 테두리가 빨간색으로 강조됩니다
  - 작은 카메라 미리보기 창은 손 상태 확인 + 키 입력용으로만 사용됩니다

Single-hand gestures:
  Open Palm          — Tracking active
  Thumb + Index      — Left click
  Thumb + Middle     — Right click
  Fist + Move Up     — Scroll up
  Fist + Move Down   — Scroll down

Two-hand gesture:
  Both hands Pinch, apart  — Zoom In
  Both hands Pinch, closer — Zoom Out

조작 키 (미리보기 창):
  d       — 손 스켈레톤 디버그 시각화 ON/OFF
  q / ESC — 종료
"""
import cv2
import numpy as np
import time
import sys
import os
from typing import Optional, List

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from core.vision.camera_capture import CameraCapture
from core.vision.hand_tracker import HandTracker, HandGesture, HandData, THUMB_TIP, INDEX_TIP, PalmRubDetector
from action.click_controller import ClickController
from action.scroll_controller import ScrollController
from action.zoom_controller import ZoomController
from action import keyboard_controller as kbd
from feedback.screen_overlay import ScreenBorderOverlayV2
from feedback.gesture_status_overlay import GestureStatusOverlay

# 카메라 미리보기 크기
PREVIEW_W = 480
PREVIEW_H = 270

# 제스처 표시 문자열 / 색상
GESTURE_COLORS = {
    HandGesture.PALM:         (80,  220,  80),
    HandGesture.PINCH_INDEX:  (80,  80,  255),
    HandGesture.PINCH_MIDDLE: (80,  80,  255),
    HandGesture.FIST:         (0,  165,  255),
    HandGesture.NONE:         (100, 100, 100),
}
GESTURE_LABELS = {
    HandGesture.PALM:         "PALM",
    HandGesture.PINCH_INDEX:  "PINCH-L",
    HandGesture.PINCH_MIDDLE: "PINCH-R",
    HandGesture.FIST:         "FIST",
    HandGesture.NONE:         "--",
}


# ── 미리보기 창 HUD ───────────────────────────────────────────────

def draw_preview_hud(frame, left: Optional[HandData], right: Optional[HandData],
                     zoom_active: bool, debug_on: bool,
                     cam_w: int, cam_h: int) -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (cam_w, 28), (18, 18, 18), -1)

    def hand_label(hand: Optional[HandData], side: str, x: int):
        if hand is None:
            cv2.putText(frame, f"{side}: --", (x, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1)
            return
        g     = hand.gesture
        color = GESTURE_COLORS.get(g, (120, 120, 120))
        label = f"{side}: {GESTURE_LABELS.get(g, '?')}"
        cv2.putText(frame, label, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

    hand_label(left,  "L", 8)
    hand_label(right, "R", 130)

    if zoom_active:
        cv2.putText(frame, "ZOOM", (cam_w // 2 - 22, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)

    dbg_color = (80, 180, 255) if debug_on else (70, 70, 70)
    cv2.putText(frame, f"Debug:{'ON' if debug_on else 'OFF'}",
                (cam_w - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, dbg_color, 1)

    hint = "d=debug  q=quit"
    cv2.putText(frame, hint, (6, cam_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (65, 65, 65), 1)
    return frame


def draw_zoom_line(frame, left: HandData, right: HandData,
                   zoom_delta: float, cam_w: int, cam_h: int) -> np.ndarray:
    """양손 핀치 연결선 (미리보기 창 내)"""
    h, w = frame.shape[:2]

    def tip_px(hand: HandData):
        lms = hand.landmarks
        mx  = (lms[THUMB_TIP].x + lms[INDEX_TIP].x) / 2
        my  = (lms[THUMB_TIP].y + lms[INDEX_TIP].y) / 2
        return int(mx * w), int(my * h)

    lp = tip_px(left)
    rp = tip_px(right)
    color = (0, 255, 180) if zoom_delta > 0 else (0, 120, 255) if zoom_delta < 0 else (180, 180, 180)
    cv2.line(frame, lp, rp, color, 2, cv2.LINE_AA)

    mid   = ((lp[0] + rp[0]) // 2, (lp[1] + rp[1]) // 2)
    cv2.circle(frame, mid, 14, color, 1, cv2.LINE_AA)
    icon  = "+" if zoom_delta > 0 else "-" if zoom_delta < 0 else "o"
    cv2.putText(frame, icon, (mid[0] - 6, mid[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for p in [lp, rp]:
        cv2.circle(frame, p, 8, color, 2, cv2.LINE_AA)
    return frame


# ── 이벤트 메시지 타이머 ─────────────────────────────────────────

class EventMessage:
    def __init__(self, duration: float = 0.6):
        self._msg     = ""
        self._expire  = 0.0
        self._duration = duration

    def set(self, msg: str):
        self._msg    = msg
        self._expire = time.time() + self._duration

    def get(self) -> str:
        return self._msg if time.time() < self._expire else ""


# ── 메인 루프 ────────────────────────────────────────────────────

def run(debug: bool = False):
    config  = load_config()
    cam_cfg = config.get("camera", {})

    # ── NSApplication 초기화 ─────────────────────────────────────
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    # ── 카메라 ──────────────────────────────────────────────────
    camera = CameraCapture(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    )
    if not camera.start():
        print("[ERROR] Camera failed to start")
        sys.exit(1)

    # ── 손 추적 + 제어 ───────────────────────────────────────────
    hand_tracker = HandTracker(config)
    click_ctrl   = ClickController(cooldown_ms=600)
    scroll_ctrl  = ScrollController()
    zoom_ctrl    = ZoomController()
    rub_detector = PalmRubDetector()
    event_msg    = EventMessage()

    # ── 오버레이 (메인 스레드) ────────────────────────────────────
    border_overlay  = ScreenBorderOverlayV2(border_width=5)
    border_overlay.start()

    gesture_overlay = GestureStatusOverlay()
    gesture_overlay.start()

    # ── 상태 ────────────────────────────────────────────────────
    prev: dict[str, HandGesture] = {"Left": HandGesture.NONE, "Right": HandGesture.NONE}
    zoom_active = False
    zoom_delta  = 0.0
    show_debug  = True   # 미리보기 창 스켈레톤 기본 ON

    print("\n" + "=" * 56)
    print("  OpenPilot — Step 2  Real Screen Hand Gesture")
    print("=" * 56)
    print("  [Single hand]")
    print("  Open Palm        -> Tracking")
    print("  Thumb + Index    -> Left Click")
    print("  Thumb + Middle   -> Right Click")
    print("  Fist + Move      -> Scroll  (fast = inertia)")
    print("  [Two hands]")
    print("  Both Pinch apart -> Zoom In")
    print("  Both Pinch close -> Zoom Out")
    print("  Both Palm + Rub  -> Close Window  (Cmd+W)")
    print("  d                -> Toggle debug skeleton")
    print("  q / ESC          -> Quit")
    print("=" * 56)
    print()
    print("  ★ Gesture events are shown on your actual screen.")
    print()

    while True:
        frame = camera.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]

        # ── 양손 인식 ─────────────────────────────────────────
        hands: List[HandData] = hand_tracker.process_multi(frame)
        by_side = {h.handedness: h for h in hands}
        left    = by_side.get("Left")
        right   = by_side.get("Right")

        curr: dict[str, HandGesture] = {
            "Left":  left.gesture  if left  else HandGesture.NONE,
            "Right": right.gesture if right else HandGesture.NONE,
        }

        # ── 양손 핀치 줌 ─────────────────────────────────────
        both_pinching = (
            curr["Left"]  == HandGesture.PINCH_INDEX and
            curr["Right"] == HandGesture.PINCH_INDEX
        )
        zoom_delta = 0.0

        if both_pinching:
            if not zoom_active:
                zoom_ctrl.start(left, right)
                zoom_active = True
                gesture_overlay.show("ZOOM START", "zoom")
                event_msg.set("ZOOM START")
            else:
                zoom_delta = zoom_ctrl.update(left, right)
                if abs(zoom_delta) > ZoomController.MIN_DELTA:
                    label = "ZOOM IN" if zoom_delta > 0 else "ZOOM OUT"
                    gesture_overlay.show(label, "zoom")
                    event_msg.set(label)
        else:
            if zoom_active:
                zoom_ctrl.stop()
                zoom_active = False

        # ── 양손 Palm 문지르기 → 창 닫기 ────────────────────
        if not zoom_active:
            if rub_detector.update(left, right):
                kbd.press_keys(["cmd", "w"])
                gesture_overlay.show("CLOSE WINDOW", "click", duration=1.0)
                event_msg.set("CLOSE WINDOW")
                print("[Step2] Palm rub → Cmd+W (close window)")

        # ── 단일 손 제스처 (줌 중엔 비활성) ─────────────────
        if not zoom_active and not rub_detector.is_active:
            for side, hand in [("Left", left), ("Right", right)]:
                curr_g = curr[side]
                prev_g = prev[side]

                if curr_g != prev_g:
                    if prev_g == HandGesture.FIST:
                        scroll_ctrl.fist_release()

                    if curr_g == HandGesture.PINCH_INDEX:
                        if click_ctrl.left_click():
                            gesture_overlay.show("LEFT CLICK", "click")
                            event_msg.set("LEFT CLICK")
                            print(f"[Step2] Left click ({side} hand)")

                    elif curr_g == HandGesture.PINCH_MIDDLE:
                        if click_ctrl.right_click():
                            gesture_overlay.show("RIGHT CLICK", "click")
                            event_msg.set("RIGHT CLICK")
                            print(f"[Step2] Right click ({side} hand)")

                if curr_g == HandGesture.FIST and hand:
                    scroll_ctrl.fist_update(hand.hand_center[1])
                    if scroll_ctrl._vel_buf:
                        vel = scroll_ctrl._vel_buf[-1]
                        if abs(vel) > 0.05:
                            label = "SCROLL UP" if vel > 0 else "SCROLL DOWN"
                            gesture_overlay.show(label, "scroll", duration=0.4)
                            event_msg.set(label)

        prev.update(curr)

        # ── 오버레이 갱신 ────────────────────────────────────
        gesture_overlay.refresh()

        # ── 작은 미리보기 창 렌더링 ──────────────────────────
        if show_debug:
            debug_frame = frame.copy()
            for hand in hands:
                debug_frame = hand_tracker.draw_debug(debug_frame, hand)
            if zoom_active and left and right:
                debug_frame = draw_zoom_line(debug_frame, left, right,
                                              zoom_delta, cam_w, cam_h)
            preview = cv2.resize(debug_frame, (PREVIEW_W, PREVIEW_H))
        else:
            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))

        if not hands:
            msg = "Show your hand"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(preview, msg, ((PREVIEW_W - tw) // 2, PREVIEW_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 220), 1)

        # rub 진행 바 (미리보기 하단)
        if rub_detector.is_active and rub_detector.progress > 0:
            bar_total = int(PREVIEW_W * 0.5)
            bar_filled = int(bar_total * rub_detector.progress)
            bar_x = (PREVIEW_W - bar_total) // 2
            bar_y = PREVIEW_H - 22
            cv2.rectangle(preview, (bar_x, bar_y),
                          (bar_x + bar_total, bar_y + 6), (40, 40, 40), -1)
            cv2.rectangle(preview, (bar_x, bar_y),
                          (bar_x + bar_filled, bar_y + 6), (80, 200, 255), -1)
            cv2.putText(preview, "RUB TO CLOSE",
                        (bar_x, bar_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 200, 255), 1)

        # 이벤트 메시지 (미리보기 하단 중앙)
        ev = event_msg.get()
        if ev:
            (tw, _), _ = cv2.getTextSize(ev, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ex, ey = (PREVIEW_W - tw) // 2, PREVIEW_H - 36
            cv2.putText(preview, ev, (ex + 1, ey + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(preview, ev, (ex, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        preview = draw_preview_hud(preview, left, right, zoom_active,
                                    show_debug, PREVIEW_W, PREVIEW_H)

        cv2.imshow("OpenPilot — Hand Gesture  [q=quit]", preview)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[Step2] Debug: {'ON' if show_debug else 'OFF'}")

    # ── 종료 ─────────────────────────────────────────────────────
    scroll_ctrl.stop_inertia()
    zoom_ctrl.stop()
    gesture_overlay.stop()
    border_overlay.stop()
    camera.stop()
    hand_tracker.close()
    cv2.destroyAllWindows()
    print("[Step2 Real] 종료됨")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(debug=args.debug)
