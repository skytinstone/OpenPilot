"""
Step 1 — Real Screen Eye Tracking
===================================
눈 추적으로 실제 화면의 마우스를 제어합니다.

카메라 테스트 창 대신 실제 데스크톱 화면에서 동작:
  - 시선 커서가 실제 화면 위에 투명 오버레이로 표시됩니다
  - 캘리브레이션: OpenCV 전체 화면 창으로 포인트 표시
  - 작은 카메라 미리보기 창은 상태 확인 및 키 입력용으로만 사용됩니다

조작 키:
  c       — 캘리브레이션 시작 (전체 화면)
  r       — 스무딩 초기화
  d       — 눈 추적 디버그 시각화 토글 (미리보기 창)
  m       — 마우스 커서 이동 ON/OFF
  q / ESC — 종료
"""
import cv2
import numpy as np
import math
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

# 카메라 미리보기 크기 (키 입력 + 상태 확인용 작은 창)
PREVIEW_W = 480
PREVIEW_H = 270

# 캘리브레이션 전체 화면 창 이름
CAL_WIN = "OpenPilot — Calibration"


# ── OpenCV 전체 화면 캘리브레이션 렌더러 ──────────────────────

def draw_calibration_frame(sw: int, sh: int, status: dict) -> np.ndarray:
    """
    캘리브레이션 상태 → 전체 화면 프레임 (OpenCV BGR).
    NSWindow 대신 OpenCV fullscreen 창에 표시.
    """
    frame = np.zeros((sh, sw, 3), dtype=np.uint8)
    phase = status.get("phase", "prepare")

    if phase == "prepare":
        _draw_prepare(frame, sw, sh, status)
        return frame

    if phase == "result":
        _draw_result(frame, sw, sh, status)
        return frame

    # 포인트 좌표
    tx = int(status.get("target_x", 0.5) * sw)
    ty = int(status.get("target_y", 0.5) * sh)
    progress     = status.get("progress", 0.0)
    is_stable    = status.get("is_stable", False)
    is_done      = (phase == "done_pt")
    is_val       = status.get("is_validation", False)
    idx          = status.get("current_idx", 0)
    total        = status.get("total", 9)
    stable_secs  = status.get("stable_seconds", 0.0)
    stable_req   = status.get("stable_required", 3.0)
    qualities    = status.get("point_qualities", [])
    confirm_time = status.get("confirm_time", 0.0)

    # 색상
    if is_val:
        color = (255, 150, 50)     # 파랑 (검증)
    elif is_done:
        color = (60, 255, 120)     # 초록 (완료)
    elif is_stable:
        color = (60, 255, 120)     # 초록 (안정)
    else:
        color = (255, 255, 255)    # 흰색 (대기)

    r_outer = 34 if not is_done else 22

    # ── 외곽 원 ──────────────────────────────────
    cv2.circle(frame, (tx, ty), r_outer, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), r_outer, color, 2, cv2.LINE_AA)

    # ── 진행 호 (arc) ────────────────────────────
    if progress > 0.01 and not is_val:
        end_angle = int(360 * progress)
        cv2.ellipse(frame, (tx, ty), (r_outer - 5, r_outer - 5),
                    -90, 0, end_angle, color, 6, cv2.LINE_AA)

    # ── 중앙 점 ──────────────────────────────────
    dot_r = 5 if not is_done else 3
    cv2.circle(frame, (tx, ty), dot_r, color, -1, cv2.LINE_AA)

    # ── done_pt: 체크마크 + 리플 ─────────────────
    if is_done and confirm_time > 0:
        elapsed = time.time() - confirm_time

        # 리플 링 (3개)
        for i in range(3):
            t = elapsed - i * 0.15
            if 0 < t < 0.9:
                frac = t / 0.9
                ring_r = int(34 + frac * 80)
                alpha = 1.0 - frac
                c = tuple(int(v * alpha) for v in color)
                cv2.circle(frame, (tx, ty), ring_r, c, 2, cv2.LINE_AA)

        # 체크마크 ✓
        if elapsed < 0.15:
            scale = elapsed / 0.15 * 1.3
        elif elapsed < 0.35:
            scale = 1.3 - (elapsed - 0.15) / 0.2 * 0.3
        else:
            scale = 1.0

        if scale > 0.05:
            sz = int(22 * scale)
            pts = np.array([
                [tx - int(sz * 0.55), ty],
                [tx - int(sz * 0.1),  ty + int(sz * 0.5)],
                [tx + int(sz * 0.6),  ty - int(sz * 0.55)],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], False, (60, 255, 120),
                          max(2, int(3.5 * scale)), cv2.LINE_AA)

        # 품질 별
        if elapsed > 0.2 and qualities:
            q = qualities[-1]
            stars = "***" if q > 0.7 else "**" if q > 0.4 else "*"
            cv2.putText(frame, stars, (tx - 15, ty + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

    # ── 안정 표시 (포인트 위) ────────────────────
    if not is_val and not is_done:
        if is_stable:
            label = f"STABLE  {stable_secs:.1f}s / {stable_req:.0f}s"
            lc = (60, 255, 120)
        else:
            label = "Look here and hold still..."
            lc = (180, 180, 180)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, label, (tx - tw // 2, ty - r_outer - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 1, cv2.LINE_AA)

    # ── 하단 안내 텍스트 ─────────────────────────
    if is_val:
        msg = f"Validation {idx - total + 1} / 4"
    elif is_done:
        msg = f"Point {idx + 1} / {total} complete!"
    else:
        msg = f"Point {idx + 1} / {total}  —  Gaze at the dot and hold still"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, msg, ((sw - tw) // 2, sh - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    # ── 포인트 품질 히스토리 바 ───────────────────
    if qualities:
        bw, gap = 24, 6
        total_w = len(qualities) * (bw + gap)
        start_x = (sw - total_w) // 2
        for i, q in enumerate(qualities):
            bx = start_x + i * (bw + gap)
            c = (60, 255, 120) if q > 0.7 else (60, 220, 255) if q > 0.4 else (60, 60, 255)
            filled = max(4, int(18 * q))
            cv2.rectangle(frame, (bx, sh - 22), (bx + bw, sh - 22 + filled), c, -1)
            cv2.rectangle(frame, (bx, sh - 22), (bx + bw, sh - 4), (80, 80, 80), 1)

    return frame


def _draw_prepare(frame, sw, sh, status):
    """준비 화면"""
    total = status.get("total", 9)
    lines = [
        ("Eye Calibration", 1.0, (255, 255, 255)),
        ("", 0, None),
        (f"{total}-Point Calibration", 0.7, (200, 220, 255)),
        ("Look at each dot and hold still for 3 seconds.", 0.55, (180, 180, 180)),
        ("Keep your head still — only move your eyes.", 0.55, (180, 180, 180)),
        ("", 0, None),
        ("Starting in a moment...", 0.5, (120, 230, 120)),
    ]
    y = sh // 2 - len(lines) * 20
    for text, scale, color in lines:
        if not text:
            y += 15
            continue
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(frame, text, ((sw - tw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += th + 18


def _draw_result(frame, sw, sh, status):
    """결과 화면"""
    acc = status.get("accuracy_px", -1)
    qualities = status.get("point_qualities", [])
    if acc > 0:
        if acc < 50:
            grade, gc = "Excellent!", (60, 255, 120)
        elif acc < 100:
            grade, gc = "Good", (60, 220, 255)
        else:
            grade, gc = "Fair — press c to retry", (60, 60, 255)
        acc_str = f"{acc:.0f} px"
    else:
        acc_str, gc, grade = "--", (180, 180, 180), ""

    avg_q = sum(qualities) / len(qualities) if qualities else 0

    lines = [
        ("Calibration Complete!", 1.0, (255, 255, 255)),
        ("", 0, None),
        (f"Accuracy: {acc_str}  {grade}", 0.7, gc),
        (f"Avg Quality: {avg_q:.0%}", 0.6, (200, 200, 255)),
        ("", 0, None),
        ("Calibration saved. Press c to recalibrate.", 0.5, (120, 120, 120)),
    ]
    y = sh // 2 - 80
    for text, scale, color in lines:
        if not text:
            y += 15
            continue
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(frame, text, ((sw - tw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += th + 18


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
    border_overlay = ScreenBorderOverlayV2(border_width=5, mode="real")
    border_overlay.start()

    gaze_cursor = GazeCursorOverlay()
    gaze_cursor.start()

    # v3.2: OpenCV 전체 화면 캘리브레이션 창 (NSWindow 대신 — 확실히 작동)
    cal_win_created = False

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

    # ── 항상 캘리브레이션부터 시작 ──────────────────────────────
    print("[Info] 9-Point 캘리브레이션 시작!")
    print("[Info] 전체 화면에 포인트가 표시됩니다. 각 점을 3초간 응시하세요.")
    mouse_enabled = False
    gaze_estimator.start_calibration(n_points=9)

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

                # v3.2: OpenCV 전체 화면 캘리브레이션 렌더링
                if not cal_win_created:
                    cv2.namedWindow(CAL_WIN, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(CAL_WIN, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                    cal_win_created = True

                cal_frame = draw_calibration_frame(screen_w, screen_h, cal_status)
                cv2.imshow(CAL_WIN, cal_frame)

                if cal_status["done"]:
                    cv2.destroyWindow(CAL_WIN)
                    cal_win_created = False
                    mouse_enabled = True
                    print("[Info] 캘리브레이션 완료 — 마우스 제어 복원")
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
        # 마우스 이동으로 OpenCV 창 포커스가 빠질 수 있으므로
        # 캘리브레이션 시작 시 마우스 제어를 일시 정지
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            mouse_enabled = False
            gaze_estimator.start_calibration(n_points=9)
            print("[Info] 9-Point 캘리브레이션 시작 — 전체 화면에 포인트 표시")
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
    if cal_win_created:
        cv2.destroyWindow(CAL_WIN)
    gaze_cursor.stop()
    border_overlay.stop()
    camera.stop()
    eye_tracker.close()
    cv2.destroyAllWindows()
    print("[Step1 Real] 종료됨")


if __name__ == "__main__":
    run()
