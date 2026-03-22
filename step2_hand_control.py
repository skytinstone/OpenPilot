"""
Step 2 — 손 제스처 기반 클릭 & 스크롤
======================================
손 제스처로 클릭과 스크롤을 제어합니다.

제스처:
  손바닥 펼침           — 트래킹 활성 (초록 아웃라인)
  엄지 + 검지 핀치      — 좌클릭   (파란 선)
  엄지 + 중지 핀치      — 우클릭   (빨간 선)
  주먹 + 위로 이동      — 위로 스크롤
  주먹 + 아래로 이동    — 아래로 스크롤
  주먹 + 빠르게 스와이프— 관성 스크롤

조작 키:
  q / ESC — 종료
  d       — 디버그 시각화 토글
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


# ── 제스처별 색상 ────────────────────────────────────────────────
GESTURE_COLORS = {
    HandGesture.PALM:          (80,  220, 80),    # 초록
    HandGesture.PINCH_INDEX:   (80,  80,  255),   # 파랑
    HandGesture.PINCH_MIDDLE:  (80,  80,  255),   # 파랑 (우클릭도 파랑 계열)
    HandGesture.FIST:          (0,   165, 255),   # 주황
    HandGesture.NONE:          (120, 120, 120),   # 회색
}

GESTURE_LABELS = {
    HandGesture.PALM:          "PALM — 트래킹 중",
    HandGesture.PINCH_INDEX:   "PINCH — 좌클릭",
    HandGesture.PINCH_MIDDLE:  "PINCH — 우클릭",
    HandGesture.FIST:          "FIST — 스크롤",
    HandGesture.NONE:          "손 인식 중...",
}


# ── 피드백 드로잉 ────────────────────────────────────────────────

def draw_gesture_hud(frame, gesture: HandGesture, event_msg: str,
                     inertia_vel: float, cam_w: int, cam_h: int) -> np.ndarray:
    """상단 상태바 + 이벤트 메시지"""
    # 상태바 배경
    cv2.rectangle(frame, (0, 0), (cam_w, 32), (25, 25, 25), -1)

    color = GESTURE_COLORS.get(gesture, (120, 120, 120))
    label = GESTURE_LABELS.get(gesture, "")

    # 제스처 이름
    cv2.putText(frame, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 관성 속도 표시
    if abs(inertia_vel) > 0.01:
        vel_str = f"관성: {inertia_vel:+.2f}"
        cv2.putText(frame, vel_str, (cam_w - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # 조작 키 힌트
    hint = "d=디버그  q=종료"
    cv2.putText(frame, hint, (cam_w // 2 - 60, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    # 이벤트 메시지 (클릭/스크롤 발생 시 중앙 하단에 표시)
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
    """손 미감지 안내"""
    msg = "손을 카메라 앞에 가져다 주세요"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, msg, ((cam_w - tw) // 2, cam_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    return frame


# ── 이벤트 메시지 타이머 ────────────────────────────────────────

class EventMessage:
    """클릭/스크롤 발생 시 화면에 잠깐 표시"""
    def __init__(self, duration: float = 0.6):
        self._msg = ""
        self._expire = 0.0
        self._duration = duration

    def set(self, msg: str):
        self._msg = msg
        self._expire = time.time() + self._duration

    def get(self) -> str:
        if time.time() < self._expire:
            return self._msg
        return ""


# ── 메인 루프 ────────────────────────────────────────────────────

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
        print("[ERROR] 카메라 시작 실패")
        sys.exit(1)

    hand_tracker  = HandTracker(config)
    click_ctrl    = ClickController(cooldown_ms=600)
    scroll_ctrl   = ScrollController()
    event_msg     = EventMessage(duration=0.6)

    # 제스처 상태 추적 (이벤트 "진입" 감지용)
    prev_gesture  = HandGesture.NONE
    show_debug    = debug

    print("\n" + "=" * 50)
    print("  Open Pilot — Step 2  손 제스처 클릭 & 스크롤")
    print("=" * 50)
    print("  손바닥      → 트래킹 활성")
    print("  엄지+검지   → 좌클릭")
    print("  엄지+중지   → 우클릭")
    print("  주먹+이동   → 스크롤  (빠르면 관성 스크롤)")
    print("  d           → 디버그 시각화")
    print("  q / ESC     → 종료")
    print("=" * 50 + "\n")

    while True:
        frame = camera.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]
        hand_data = hand_tracker.process(frame)

        curr_gesture = hand_data.gesture if hand_data else HandGesture.NONE

        # ── 제스처 상태 전이 처리 ─────────────────────────────
        if curr_gesture != prev_gesture:

            # 주먹 해제 → 관성 발동
            if prev_gesture == HandGesture.FIST:
                scroll_ctrl.fist_release()

            # 핀치 진입 → 클릭 (on-enter 방식으로 1회만)
            if curr_gesture == HandGesture.PINCH_INDEX:
                if click_ctrl.left_click():
                    event_msg.set("👆 LEFT CLICK")
                    print("[Step2] 좌클릭")

            elif curr_gesture == HandGesture.PINCH_MIDDLE:
                if click_ctrl.right_click():
                    event_msg.set("✌️ RIGHT CLICK")
                    print("[Step2] 우클릭")

        # ── 주먹 지속 → 직접 스크롤 ──────────────────────────
        if curr_gesture == HandGesture.FIST and hand_data:
            scroll_ctrl.fist_update(hand_data.hand_center[1])
            vel = scroll_ctrl.inertia_velocity
            # 스크롤 방향 피드백 (직접 이동 중에도 표시)
            if abs(scroll_ctrl._vel_buf[-1]) > 0.05 if scroll_ctrl._vel_buf else False:
                direction = "▲ 스크롤 위" if scroll_ctrl._vel_buf[-1] > 0 else "▼ 스크롤 아래"
                event_msg.set(direction)

        prev_gesture = curr_gesture

        # ── 디버그 시각화 ─────────────────────────────────────
        if show_debug and hand_data:
            frame = hand_tracker.draw_debug(frame, hand_data)

        # ── HUD 그리기 ────────────────────────────────────────
        if hand_data is None:
            frame = draw_no_hand(frame, cam_w, cam_h)

        frame = draw_gesture_hud(
            frame, curr_gesture, event_msg.get(),
            scroll_ctrl.inertia_velocity, cam_w, cam_h,
        )

        cv2.imshow("Open Pilot — Step 2  Hand Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[Step2] 디버그 시각화: {'ON' if show_debug else 'OFF'}")

    # ── 종료 ──────────────────────────────────────────────────
    scroll_ctrl.stop_inertia()
    camera.stop()
    hand_tracker.close()
    cv2.destroyAllWindows()
    print("[Step2] 종료됨")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(debug=args.debug)
