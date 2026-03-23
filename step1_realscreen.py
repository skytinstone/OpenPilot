"""
Step 1 — Real Screen Eye Tracking
===================================
눈 추적으로 실제 화면의 마우스를 제어합니다.

카메라 테스트 창 대신 실제 데스크톱 화면에서 동작:
  - 시선 커서가 실제 화면 위에 투명 오버레이로 표시됩니다
  - 캘리브레이션이 실제 화면 전체에 오버레이로 표시됩니다
  - 작은 카메라 미리보기 창은 상태 확인 및 키 입력용으로만 사용됩니다

조작 키:
  c       — 캘리브레이션 시작 (실제 화면에 포인트 표시)
  r       — 스무딩 초기화
  d       — 눈 추적 디버그 시각화 토글 (미리보기 창)
  m       — 마우스 커서 이동 ON/OFF
  q / ESC — 종료
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
from core.vision.eye_tracker import EyeTracker
from core.vision.gaze_estimator import GazeEstimator
from feedback.screen_overlay import ScreenBorderOverlayV2
from feedback.gaze_cursor_overlay import GazeCursorOverlay
from feedback.calibration_screen_overlay import CalibrationScreenOverlay

# 카메라 미리보기 크기 (키 입력 + 상태 확인용 작은 창)
PREVIEW_W = 480
PREVIEW_H = 270


def get_screen_size(config: dict = None):
    if config:
        scr = config.get("screen", {})
        w, h = scr.get("width"), scr.get("height")
        if w and h:
            return int(w), int(h)
    try:
        from AppKit import NSScreen
        frame = NSScreen.mainScreen().frame()
        return int(frame.size.width), int(frame.size.height)
    except Exception:
        return 1440, 900


def draw_preview_hud(frame, fps: float, is_calibrated: bool,
                     mouse_enabled: bool, eye_detected: bool,
                     debug_on: bool, cam_w: int, cam_h: int) -> np.ndarray:
    """작은 미리보기 창 상단 상태 표시"""
    cv2.rectangle(frame, (0, 0), (cam_w, 28), (20, 20, 20), -1)

    items = [
        (f"FPS {fps:.0f}",                             (140, 140, 140)),
        (f"Eye: {'OK' if eye_detected else '--'}",      (80, 220, 80) if eye_detected else (80, 80, 220)),
        (f"Cal: {'done' if is_calibrated else 'c=go'}", (80, 220, 80) if is_calibrated else (200, 180, 60)),
        (f"Mouse: {'ON' if mouse_enabled else 'OFF'}",  (80, 220, 80) if mouse_enabled else (120, 120, 120)),
        (f"Debug: {'ON' if debug_on else 'OFF'}",       (80, 180, 255) if debug_on else (80, 80, 80)),
    ]
    x = 6
    for text, color in items:
        cv2.putText(frame, text, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 14

    # 하단 키 힌트
    hint = "c=cal  r=reset  m=mouse  d=debug  q=quit"
    cv2.putText(frame, hint, (6, cam_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (65, 65, 65), 1)
    return frame


def run():
    config   = load_config()
    screen_w, screen_h = get_screen_size(config)
    print(f"[Info] Screen: {screen_w} × {screen_h}")

    # ── NSApplication 초기화 (오버레이보다 먼저) ─────────────────
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    # ── 카메라 ──────────────────────────────────────────────────
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

    # ── 눈 추적 + 시선 추정 ──────────────────────────────────────
    eye_tracker    = EyeTracker(config)
    gaze_estimator = GazeEstimator(screen_w, screen_h, config)

    # ── 오버레이들 (메인 스레드에서 생성) ───────────────────────
    border_overlay = ScreenBorderOverlayV2(border_width=5)
    border_overlay.start()

    gaze_cursor = GazeCursorOverlay()
    gaze_cursor.start()

    cal_overlay = CalibrationScreenOverlay()
    cal_overlay.start()

    # ── 마우스 컨트롤러 ─────────────────────────────────────────
    mouse         = None
    mouse_enabled = True
    try:
        from action.mouse_controller import MouseController
        mouse = MouseController()
    except Exception as e:
        print(f"[WARN] MouseController: {e}")
        mouse_enabled = False

    # ── 상태 변수 ────────────────────────────────────────────────
    show_debug   = False
    fps_counter  = 0
    fps_start    = time.time()
    current_fps  = 0.0
    eye_detected = False
    gaze_point   = None

    print("\n" + "=" * 54)
    print("  OpenPilot — Step 1  Real Screen Mode")
    print("=" * 54)
    print("  c       : Calibration  (overlaid on real screen)")
    print("  d       : Toggle eye tracking debug in preview")
    print("  m       : Toggle mouse control")
    print("  r       : Reset smoothing")
    print("  q / ESC : Quit")
    print("=" * 54 + "\n")
    print("  ★ Gaze cursor is shown on your actual screen.")
    print("  ★ Calibration points will appear on your actual screen.")
    print()

    while True:
        frame = camera.read()
        if frame is None:
            continue

        # ── FPS ───────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        # ── 눈 추적 ──────────────────────────────────────────
        eye_data     = eye_tracker.process(frame)
        eye_detected = eye_data is not None

        # ── 시선 추정 + 캘리브레이션 ─────────────────────────
        if eye_data is not None:
            if gaze_estimator.is_calibrating:
                cal_status = gaze_estimator.update_calibration(eye_data)
                # 현재 포인트 좌표를 status에 추가
                cal_pt = gaze_estimator.current_calibration_point
                val_pt = gaze_estimator.current_validation_point
                if val_pt:
                    cal_status["target_x"] = val_pt[0]
                    cal_status["target_y"] = val_pt[1]
                elif cal_pt:
                    cal_status["target_x"] = cal_pt.screen_x
                    cal_status["target_y"] = cal_pt.screen_y
                cal_overlay.update_from_status(cal_status)
                if cal_status["done"]:
                    cal_overlay.hide()
            else:
                gaze_point = gaze_estimator.estimate(eye_data)
                if gaze_point:
                    if mouse_enabled and mouse:
                        mouse.move(gaze_point.x, gaze_point.y)
                    gaze_cursor.update(gaze_point.x, gaze_point.y)
                else:
                    gaze_cursor.hide_cursor()
        else:
            gaze_cursor.hide_cursor()

        # ── 작은 미리보기 창 렌더링 ──────────────────────────
        if show_debug and eye_data is not None:
            gx = gaze_point.x if gaze_point else -1
            gy = gaze_point.y if gaze_point else -1
            debug_frame = eye_tracker.draw_debug(
                frame.copy(), eye_data,
                gaze_x=gx, gaze_y=gy,
                screen_w=screen_w, screen_h=screen_h,
            )
            preview = cv2.resize(debug_frame, (PREVIEW_W, PREVIEW_H))
        else:
            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))

        preview = draw_preview_hud(
            preview, current_fps, gaze_estimator.is_calibrated,
            mouse_enabled, eye_detected, show_debug,
            PREVIEW_W, PREVIEW_H,
        )

        cv2.imshow("OpenPilot — Eye Tracking  [q=quit]", preview)
        key = cv2.waitKey(1) & 0xFF

        # ── 키 입력 ───────────────────────────────────────────
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            gaze_estimator.start_calibration(n_points=9)
            print("[Info] 9-Point 캘리브레이션 시작 — 실제 화면에 포인트가 표시됩니다")
        elif key == ord('r'):
            gaze_estimator.reset()
            print("[Info] 스무딩 초기화")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[Info] Debug: {'ON' if show_debug else 'OFF'}")
        elif key == ord('m'):
            mouse_enabled = not mouse_enabled
            if mouse_enabled and mouse is None:
                try:
                    from action.mouse_controller import MouseController
                    mouse = MouseController()
                except Exception:
                    mouse_enabled = False
            print(f"[Info] Mouse: {'ON' if mouse_enabled else 'OFF'}")

    # ── 종료 ─────────────────────────────────────────────────────
    cal_overlay.stop()
    gaze_cursor.stop()
    border_overlay.stop()
    camera.stop()
    eye_tracker.close()
    cv2.destroyAllWindows()
    print("[Step1 Real] 종료됨")


if __name__ == "__main__":
    run()
