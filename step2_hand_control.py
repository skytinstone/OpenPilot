"""
Step 2 - Hand Gesture Click, Scroll & Zoom
===========================================
Single-hand gestures:
  Open Palm          - Tracking active
  Thumb + Index      - Left click
  Thumb + Middle     - Right click
  Fist + Move Up     - Scroll up
  Fist + Move Down   - Scroll down
  Fist + Fast Swipe  - Inertia scroll

Two-hand gesture:
  Both hands Thumb+Index pinch, move apart  - Zoom In
  Both hands Thumb+Index pinch, move closer - Zoom Out

Keys:
  d       - Toggle debug skeleton
  q / ESC - Quit
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


# ── Gesture display ──────────────────────────────────────────────
GESTURE_COLORS = {
    HandGesture.PALM:         (80,  220, 80),
    HandGesture.PINCH_INDEX:  (80,  80,  255),
    HandGesture.PINCH_MIDDLE: (80,  80,  255),
    HandGesture.FIST:         (0,   165, 255),
    HandGesture.NONE:         (120, 120, 120),
}
GESTURE_LABELS = {
    HandGesture.PALM:         "PALM",
    HandGesture.PINCH_INDEX:  "PINCH-L",
    HandGesture.PINCH_MIDDLE: "PINCH-R",
    HandGesture.FIST:         "FIST",
    HandGesture.NONE:         "-",
}


# ── Drawing helpers ──────────────────────────────────────────────

def draw_hud(frame, left: Optional[HandData], right: Optional[HandData],
             zoom_active: bool, zoom_delta: float,
             event_msg: str, cam_w: int, cam_h: int) -> np.ndarray:
    """Top status bar with per-hand state + event overlay"""

    # 상태바 배경
    cv2.rectangle(frame, (0, 0), (cam_w, 36), (20, 20, 20), -1)

    def hand_label(hand: Optional[HandData], side: str, x: int):
        if hand is None:
            cv2.putText(frame, f"{side}: --", (x, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1)
            return
        g = hand.gesture
        color = GESTURE_COLORS.get(g, (120, 120, 120))
        label = f"{side}: {GESTURE_LABELS.get(g, '?')}"
        cv2.putText(frame, label, (x, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    hand_label(left,  "L", 10)
    hand_label(right, "R", 140)

    # 줌 상태 표시
    if zoom_active:
        zoom_str = f"ZOOM {'IN' if zoom_delta > 0 else 'OUT' if zoom_delta < 0 else '...'}"
        cv2.putText(frame, zoom_str, (cam_w // 2 - 50, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # 도움말
    cv2.putText(frame, "d=debug  q=quit", (cam_w - 150, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # 이벤트 메시지 (하단 중앙)
    if event_msg:
        (tw, _), _ = cv2.getTextSize(event_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        ex, ey = (cam_w - tw) // 2, cam_h - 24
        cv2.putText(frame, event_msg, (ex+1, ey+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(frame, event_msg, (ex, ey),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

    return frame


def draw_no_hand(frame, cam_w: int, cam_h: int) -> np.ndarray:
    msg = "Show your hand to the camera"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, msg, ((cam_w - tw) // 2, cam_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2)
    return frame


def draw_zoom_line(frame, left: HandData, right: HandData,
                   zoom_delta: float, cam_w: int, cam_h: int) -> np.ndarray:
    """양손 핀치 사이 연결선 + 줌 방향 표시"""
    h, w = frame.shape[:2]

    def tip_px(hand: HandData):
        lms = hand.landmarks
        mx = (lms[THUMB_TIP].x + lms[INDEX_TIP].x) / 2
        my = (lms[THUMB_TIP].y + lms[INDEX_TIP].y) / 2
        return int(mx * w), int(my * h)

    lp = tip_px(left)
    rp = tip_px(right)

    # 연결선
    color = (0, 255, 180) if zoom_delta > 0 else (0, 120, 255) if zoom_delta < 0 else (180, 180, 180)
    cv2.line(frame, lp, rp, color, 2, cv2.LINE_AA)

    # 중점에 줌 방향 아이콘
    mid = ((lp[0] + rp[0]) // 2, (lp[1] + rp[1]) // 2)
    cv2.circle(frame, mid, 18, color, 1, cv2.LINE_AA)
    icon = "+" if zoom_delta > 0 else "-" if zoom_delta < 0 else "o"
    cv2.putText(frame, icon, (mid[0] - 7, mid[1] + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 거리 수치
    dist = ((lp[0]-rp[0])**2 + (lp[1]-rp[1])**2) ** 0.5
    cv2.putText(frame, f"{dist:.0f}px", (mid[0] - 20, mid[1] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # 핀치 포인트 강조
    for p in [lp, rp]:
        cv2.circle(frame, p, 10, color, 2, cv2.LINE_AA)

    return frame


# ── Event message timer ──────────────────────────────────────────

class EventMessage:
    def __init__(self, duration: float = 0.7):
        self._msg = ""
        self._expire = 0.0
        self._duration = duration

    def set(self, msg: str):
        self._msg = msg
        self._expire = time.time() + self._duration

    def get(self) -> str:
        return self._msg if time.time() < self._expire else ""


# ── Main loop ────────────────────────────────────────────────────

def run(debug: bool = False):
    config   = load_config()
    cam_cfg  = config.get("camera", {})

    camera = CameraCapture(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    )
    if not camera.start():
        print("[ERROR] Camera failed to start")
        sys.exit(1)

    hand_tracker = HandTracker(config)
    click_ctrl   = ClickController(cooldown_ms=600)
    scroll_ctrl  = ScrollController()
    zoom_ctrl    = ZoomController()
    rub_detector = PalmRubDetector()
    event_msg    = EventMessage()

    # ── 좌/우 손 각각의 이전 제스처 상태 ──────────────────────────
    prev: dict[str, HandGesture] = {"Left": HandGesture.NONE, "Right": HandGesture.NONE}
    zoom_active  = False
    zoom_delta   = 0.0
    show_debug   = True   # skeleton ON by default

    print("\n" + "=" * 56)
    print("  Open Pilot - Step 2  Hand Gesture Control")
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
    print("=" * 56 + "\n")

    while True:
        frame = camera.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]

        # ── 양손 인식 ─────────────────────────────────────────────
        hands: List[HandData] = hand_tracker.process_multi(frame)
        by_side = {h.handedness: h for h in hands}

        left  = by_side.get("Left")
        right = by_side.get("Right")

        curr: dict[str, HandGesture] = {
            "Left":  left.gesture  if left  else HandGesture.NONE,
            "Right": right.gesture if right else HandGesture.NONE,
        }

        # ── 양손 핀치 줌 감지 ────────────────────────────────────
        both_pinching = (
            curr["Left"]  == HandGesture.PINCH_INDEX and
            curr["Right"] == HandGesture.PINCH_INDEX
        )

        zoom_delta = 0.0

        if both_pinching:
            if not zoom_active:
                zoom_ctrl.start(left, right)
                zoom_active = True
                event_msg.set("ZOOM START")
            else:
                zoom_delta = zoom_ctrl.update(left, right)
                if abs(zoom_delta) > ZoomController.MIN_DELTA:
                    event_msg.set("ZOOM IN" if zoom_delta > 0 else "ZOOM OUT")
        else:
            if zoom_active:
                zoom_ctrl.stop()
                zoom_active = False

        # ── 양손 Palm 문지르기 → 창 닫기 ─────────────────────────
        if not zoom_active:
            if rub_detector.update(left, right):
                kbd.press_keys(["cmd", "w"])
                event_msg.set("CLOSE WINDOW")
                print("[Step2] Palm rub → Cmd+W (close window)")

        # ── 단일 손 제스처 처리 (줌 중엔 클릭/스크롤 비활성) ────────
        if not zoom_active and not rub_detector.is_active:
            for side, hand in [("Left", left), ("Right", right)]:
                curr_g = curr[side]
                prev_g = prev[side]

                if curr_g != prev_g:
                    # Fist 해제 → 관성 발동
                    if prev_g == HandGesture.FIST:
                        scroll_ctrl.fist_release()

                    # 핀치 진입 → 클릭 (on-enter)
                    if curr_g == HandGesture.PINCH_INDEX:
                        if click_ctrl.left_click():
                            event_msg.set("LEFT CLICK")
                            print(f"[Step2] Left click ({side} hand)")

                    elif curr_g == HandGesture.PINCH_MIDDLE:
                        if click_ctrl.right_click():
                            event_msg.set("RIGHT CLICK")
                            print(f"[Step2] Right click ({side} hand)")

                # Fist 지속 → 직접 스크롤
                if curr_g == HandGesture.FIST and hand:
                    scroll_ctrl.fist_update(hand.hand_center[1])
                    if scroll_ctrl._vel_buf:
                        vel = scroll_ctrl._vel_buf[-1]
                        if abs(vel) > 0.05:
                            event_msg.set("SCROLL UP" if vel > 0 else "SCROLL DOWN")

        # 이전 제스처 갱신
        prev.update(curr)

        # ── 시각화 ───────────────────────────────────────────────
        if show_debug:
            for hand in hands:
                frame = hand_tracker.draw_debug(frame, hand)

        # 양손 줌 연결선
        if zoom_active and left and right:
            frame = draw_zoom_line(frame, left, right, zoom_delta, cam_w, cam_h)

        if not hands:
            frame = draw_no_hand(frame, cam_w, cam_h)

        # rub 진행 표시
        if rub_detector.is_active and rub_detector.progress > 0:
            bar_w = int(cam_w * 0.4 * rub_detector.progress)
            bar_x = (cam_w - int(cam_w * 0.4)) // 2
            bar_y = cam_h - 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(cam_w * 0.4), bar_y + 8),
                          (40, 40, 40), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                          (80, 200, 255), -1)
            cv2.putText(frame, "RUB TO CLOSE",
                        (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (80, 200, 255), 1)

        frame = draw_hud(frame, left, right, zoom_active, zoom_delta,
                         event_msg.get(), cam_w, cam_h)

        cv2.imshow("Open Pilot - Step 2  Hand Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[Step2] Debug: {'ON' if show_debug else 'OFF'}")

    # ── Cleanup ────────────────────────────────────────────────
    scroll_ctrl.stop_inertia()
    zoom_ctrl.stop()
    camera.stop()
    hand_tracker.close()
    cv2.destroyAllWindows()
    print("[Step2] Stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(debug=args.debug)
