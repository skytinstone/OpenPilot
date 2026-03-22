"""
Step 2 - Hand Gesture Click & Scroll
=====================================
Gestures:
  Open Palm         - Tracking active (green outline)
  Thumb + Index     - Left click  (blue)
  Thumb + Middle    - Right click (blue)
  Fist + Move Up    - Scroll up
  Fist + Move Down  - Scroll down
  Fist + Fast Swipe - Inertia scroll

Keys:
  q / ESC - Quit
  d       - Toggle debug landmarks
"""
import cv2
import numpy as np
import time
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from core.vision.camera_capture import CameraCapture
from core.vision.hand_tracker import HandTracker, HandGesture
from action.click_controller import ClickController
from action.scroll_controller import ScrollController


# ── Gesture colors ───────────────────────────────────────────────
GESTURE_COLORS = {
    HandGesture.PALM:         (80,  220, 80),   # green
    HandGesture.PINCH_INDEX:  (80,  80,  255),  # blue
    HandGesture.PINCH_MIDDLE: (80,  80,  255),  # blue
    HandGesture.FIST:         (0,   165, 255),  # orange
    HandGesture.NONE:         (120, 120, 120),  # gray
}

GESTURE_LABELS = {
    HandGesture.PALM:         "PALM - Tracking",
    HandGesture.PINCH_INDEX:  "PINCH - Left Click",
    HandGesture.PINCH_MIDDLE: "PINCH - Right Click",
    HandGesture.FIST:         "FIST - Scroll",
    HandGesture.NONE:         "Detecting hand...",
}


# ── Drawing helpers ──────────────────────────────────────────────

def draw_gesture_hud(frame, gesture: HandGesture, event_msg: str,
                     inertia_vel: float, cam_w: int, cam_h: int) -> np.ndarray:
    """Top status bar + event message overlay"""
    cv2.rectangle(frame, (0, 0), (cam_w, 32), (25, 25, 25), -1)

    color = GESTURE_COLORS.get(gesture, (120, 120, 120))
    label = GESTURE_LABELS.get(gesture, "")

    cv2.putText(frame, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if abs(inertia_vel) > 0.01:
        vel_str = f"Inertia: {inertia_vel:+.2f}"
        cv2.putText(frame, vel_str, (cam_w - 180, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    hint = "d=debug  q=quit"
    cv2.putText(frame, hint, (cam_w // 2 - 60, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    if event_msg:
        (tw, th), _ = cv2.getTextSize(event_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        ex = (cam_w - tw) // 2
        ey = cam_h - 24
        cv2.putText(frame, event_msg, (ex + 1, ey + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(frame, event_msg, (ex, ey),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame


def draw_no_hand(frame, cam_w: int, cam_h: int) -> np.ndarray:
    """No hand detected guidance"""
    msg = "Show your hand to the camera"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, msg, ((cam_w - tw) // 2, cam_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    return frame


# ── Event message timer ──────────────────────────────────────────

class EventMessage:
    def __init__(self, duration: float = 0.6):
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
    config = load_config()
    cam_cfg = config.get("camera", {})

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
    event_msg    = EventMessage(duration=0.6)

    prev_gesture = HandGesture.NONE
    show_debug   = True   # skeleton visualization ON by default (toggle with 'd')

    print("\n" + "=" * 52)
    print("  Open Pilot - Step 2  Hand Gesture Control")
    print("=" * 52)
    print("  Open Palm        -> Tracking active")
    print("  Thumb + Index    -> Left Click")
    print("  Thumb + Middle   -> Right Click")
    print("  Fist + Move      -> Scroll  (fast = inertia)")
    print("  d                -> Debug landmarks")
    print("  q / ESC          -> Quit")
    print("=" * 52 + "\n")

    while True:
        frame = camera.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]
        hand_data    = hand_tracker.process(frame)
        curr_gesture = hand_data.gesture if hand_data else HandGesture.NONE

        # ── Gesture state transitions ──────────────────────────
        if curr_gesture != prev_gesture:

            # Fist released -> trigger inertia
            if prev_gesture == HandGesture.FIST:
                scroll_ctrl.fist_release()

            # Pinch enter -> click once
            if curr_gesture == HandGesture.PINCH_INDEX:
                if click_ctrl.left_click():
                    event_msg.set("LEFT CLICK")
                    print("[Step2] Left click")

            elif curr_gesture == HandGesture.PINCH_MIDDLE:
                if click_ctrl.right_click():
                    event_msg.set("RIGHT CLICK")
                    print("[Step2] Right click")

        # ── Fist active -> direct scroll ───────────────────────
        if curr_gesture == HandGesture.FIST and hand_data:
            scroll_ctrl.fist_update(hand_data.hand_center[1])
            if scroll_ctrl._vel_buf:
                vel = scroll_ctrl._vel_buf[-1]
                if abs(vel) > 0.05:
                    direction = "SCROLL UP" if vel > 0 else "SCROLL DOWN"
                    event_msg.set(direction)

        prev_gesture = curr_gesture

        # ── Draw ───────────────────────────────────────────────
        if show_debug and hand_data:
            frame = hand_tracker.draw_debug(frame, hand_data)

        if hand_data is None:
            frame = draw_no_hand(frame, cam_w, cam_h)

        frame = draw_gesture_hud(
            frame, curr_gesture, event_msg.get(),
            scroll_ctrl.inertia_velocity, cam_w, cam_h,
        )

        cv2.imshow("Open Pilot - Step 2  Hand Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[Step2] Debug: {'ON' if show_debug else 'OFF'}")

    # ── Cleanup ────────────────────────────────────────────────
    scroll_ctrl.stop_inertia()
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
