"""
Step 1 — 눈 트래킹 화면 타겟팅 테스트
======================================
눈(홍채)으로 화면의 어느 부분을 보고 있는지 실시간으로 확인하는 테스트

시각 피드백:
  - 화면 테두리: 빨간색 (Open Pilot 실행 중 표시)
  - 시선 포인터: 짙은 회색 원형
  - 눈 감기 3초 → 뜨는 순간 시선 초점 자동 재설정

조작 키:
  c       — 캘리브레이션 시작
  r       — 스무딩 상태 초기화
  d       — 디버그 시각화 토글
  m       — 실제 마우스 커서 이동 토글 (접근성 권한 필요)
  q / ESC — 종료

실행:
  python step1_targeting_test.py
  python step1_targeting_test.py --no-mouse   # 마우스 이동 없이 테스트
  python step1_targeting_test.py --debug      # 랜드마크 시각화 포함
"""
import cv2
import numpy as np
import time
import argparse
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from core.vision.camera_capture import CameraCapture
from core.vision.eye_tracker import EyeTracker
from core.vision.gaze_estimator import GazeEstimator, CALIBRATION_POSITIONS
from core.vision.hover_detector import HoverDetector
from feedback.screen_overlay import ScreenBorderOverlayV2
from feedback.hover_overlay import HoverOverlay, hover_state


# ── 눈 감기 3초 → 시선 초점 재설정 ────────────────────────────────────────
BLINK_CLOSE_THRESHOLD = 0.15   # 이 값 미만이면 눈 감음으로 판단
BLINK_RESET_DURATION  = 3.0    # 몇 초 감아야 리셋 트리거 (초)


class BlinkResetDetector:
    """
    양쪽 눈을 BLINK_RESET_DURATION 초 이상 감으면 '리셋 대기' 상태로 전환.
    그 후 눈을 뜨는 순간 gaze_estimator.reset() 을 호출해 커서 초점을 재설정.
    """

    def __init__(self, close_threshold: float = BLINK_CLOSE_THRESHOLD,
                 reset_duration: float = BLINK_RESET_DURATION):
        self._threshold  = close_threshold
        self._duration   = reset_duration
        self._close_start: Optional[float] = None
        self._reset_pending = False
        self.just_reset  = False   # 이번 프레임에 리셋됐으면 True (시각 표시용)

    def update(self, eye_data) -> bool:
        """
        매 프레임 호출. 리셋이 실행된 프레임에 True 반환.
        eye_data 가 None 이면 눈 미감지로 간주해 타이머 유지.
        """
        self.just_reset = False

        if eye_data is None:
            return False

        lo = eye_data.left_eye_openness
        ro = eye_data.right_eye_openness
        both_closed = (lo < self._threshold) and (ro < self._threshold)

        if both_closed:
            if self._close_start is None:
                self._close_start = time.time()
            elif (not self._reset_pending and
                  time.time() - self._close_start >= self._duration):
                self._reset_pending = True   # 3초 달성 → 눈 뜰 때 리셋
        else:
            # 눈 뜸
            if self._reset_pending:
                self._reset_pending = False
                self._close_start   = None
                self.just_reset     = True
                return True
            self._close_start = None

        return False

    @property
    def close_progress(self) -> float:
        """눈 감은 시간 비율 (0.0 ~ 1.0). 1.0 이면 3초 달성."""
        if self._close_start is None:
            return 0.0
        return min((time.time() - self._close_start) / self._duration, 1.0)

    @property
    def is_waiting_reset(self) -> bool:
        """3초 달성 후 눈 뜰 때를 기다리는 상태"""
        return self._reset_pending


def draw_blink_indicator(frame, detector: BlinkResetDetector, cam_w, cam_h):
    """
    눈 감기 진행 상태를 화면 중앙 하단에 표시.
    - 감는 중: 회색 프로그레스 바
    - 3초 달성 대기 중: 파란색 바 + '눈 뜨세요' 메시지
    - 리셋 직후: 초록색 플래시 메시지
    """
    from typing import Optional   # 이미 임포트돼 있지만 안전하게

    progress = detector.close_progress

    if detector.just_reset:
        # 초록 플래시
        msg = "초점 재설정!"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, msg,
                    ((cam_w - tw) // 2, cam_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 60), 2)
        return frame

    if progress <= 0.0:
        return frame   # 눈 뜬 상태, 표시 없음

    # 프로그레스 바
    bar_w  = 200
    bar_h  = 12
    bar_x  = (cam_w - bar_w) // 2
    bar_y  = cam_h - 90
    filled = int(bar_w * progress)

    color_bg   = (50, 50, 50)
    color_fill = (60, 180, 255) if detector.is_waiting_reset else (120, 120, 120)
    color_text = (60, 200, 255) if detector.is_waiting_reset else (180, 180, 180)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_bg, -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color_fill, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    if detector.is_waiting_reset:
        msg = "눈을 떠서 초점을 맞추세요"
    else:
        secs = BLINK_RESET_DURATION * (1.0 - progress)
        msg = f"눈 감는 중... ({secs:.1f}초)"

    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, msg,
                ((cam_w - tw) // 2, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)

    return frame


def get_screen_size():
    try:
        from AppKit import NSScreen
        screen = NSScreen.mainScreen()
        frame = screen.frame()
        return int(frame.size.width), int(frame.size.height)
    except Exception:
        return 1440, 900


def draw_gaze_pointer(frame, gaze_x, gaze_y, screen_w, screen_h, cam_w, cam_h):
    """
    시선 포인터: 짙은 회색 원형 포인터
    - 외곽: 반투명 짙은 회색 큰 원
    - 내부: 불투명 짙은 회색 작은 원 (중심점)
    - 좌표 텍스트: 포인터 우측 하단
    """
    # 화면 좌표 → 카메라 프레임 좌표로 역변환
    fx = int(gaze_x / screen_w * cam_w)
    fy = int(gaze_y / screen_h * cam_h)

    DARK_GRAY       = (60, 60, 60)
    DARK_GRAY_LIGHT = (90, 90, 90)
    WHITE           = (220, 220, 220)

    # 외곽 원 (반투명 효과: 오버레이 블렌딩)
    overlay = frame.copy()
    cv2.circle(overlay, (fx, fy), 22, DARK_GRAY, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # 테두리 원
    cv2.circle(frame, (fx, fy), 22, DARK_GRAY_LIGHT, 2)

    # 내부 중심 원 (불투명)
    cv2.circle(frame, (fx, fy), 7, DARK_GRAY, -1)
    cv2.circle(frame, (fx, fy), 7, DARK_GRAY_LIGHT, 1)

    # 좌표 텍스트 (포인터 우측 하단, 흰색 그림자 효과)
    label = f"({gaze_x}, {gaze_y})"
    tx, ty = fx + 16, fy + 16
    cv2.putText(frame, label, (tx + 1, ty + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    return frame


def draw_calibration_screen(frame, cal_point, progress, idx, total, cam_w, cam_h, screen_w, screen_h):
    """캘리브레이션 화면 그리기"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, cam_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 목표 포인트
    tx = int(cal_point.screen_x * cam_w)
    ty = int(cal_point.screen_y * cam_h)

    # 진행 원 (채워지는 애니메이션)
    radius = 30
    cv2.circle(frame, (tx, ty), radius, (80, 80, 80), -1)
    cv2.circle(frame, (tx, ty), radius, (255, 255, 255), 2)

    # 진행률 부채꼴
    angle = int(360 * progress)
    if angle > 0:
        cv2.ellipse(frame, (tx, ty), (radius - 4, radius - 4),
                    -90, 0, angle, (0, 255, 100), 4)

    # 중앙 점
    cv2.circle(frame, (tx, ty), 5, (255, 255, 255), -1)

    # 안내 텍스트
    text = f"이 점을 바라보세요 ({idx + 1}/{total})"
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, text, ((cam_w - tw) // 2, cam_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def draw_hover_badge(frame, hovered_element, cam_w, cam_h):
    """카메라 프리뷰 하단에 호버된 요소 이름 배지 표시"""
    if hovered_element is None:
        return frame

    title = hovered_element.title or hovered_element.role
    dwell = hovered_element.dwell_time
    progress = min(dwell / 0.8, 1.0)

    # 배지 배경
    label = f"  {title}  "
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bx = (cam_w - tw) // 2
    by = cam_h - 60
    cv2.rectangle(frame, (bx - 4, by - th - 6), (bx + tw + 4, by + 6), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx - 4, by - th - 6), (bx + tw + 4, by + 6), (80, 80, 80), 1)
    cv2.putText(frame, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    # 드웰 프로그레스 바 (배지 하단)
    bar_y = by + 8
    bar_x_start = bx - 4
    bar_w = tw + 8
    cv2.rectangle(frame, (bar_x_start, bar_y), (bar_x_start + bar_w, bar_y + 4), (50, 50, 50), -1)
    fill_w = int(bar_w * progress)
    color = (80, 220, 255) if progress < 1.0 else (80, 255, 120)
    cv2.rectangle(frame, (bar_x_start, bar_y), (bar_x_start + fill_w, bar_y + 4), color, -1)

    return frame


def draw_hud(frame, fps, is_calibrated, mouse_enabled, eye_detected, blink_progress=0.0):
    """HUD 상태 표시"""
    h, w = frame.shape[:2]

    # 상태바 배경
    cv2.rectangle(frame, (0, 0), (w, 28), (30, 30, 30), -1)

    blink_str = f"눈감기: {int(blink_progress*100)}%" if blink_progress > 0 else "눈감기: -"
    status_items = [
        f"FPS: {fps:.0f}",
        f"눈감지: {'✓' if eye_detected else '✗'}",
        f"캘리브레이션: {'완료' if is_calibrated else '미완료(c키)'}",
        f"마우스제어: {'ON(m키OFF)' if mouse_enabled else 'OFF(m키ON)'}",
        blink_str,
        "q=종료 c=캘리브 r=리셋 d=디버그",
    ]

    x = 8
    for item in status_items:
        color = (200, 200, 200)
        if "✗" in item:
            color = (80, 80, 255)
        elif "✓" in item or "완료" in item or "ON" in item:
            color = (80, 255, 80)
        cv2.putText(frame, item, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        x += cv2.getTextSize(item, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 16

    return frame


def draw_target_grid(frame, cam_w, cam_h):
    """화면 4모서리 + 중앙에 타겟 마커 표시"""
    targets = [
        (int(0.1 * cam_w), int(0.1 * cam_h)),
        (int(0.9 * cam_w), int(0.1 * cam_h)),
        (int(0.5 * cam_w), int(0.5 * cam_h)),
        (int(0.1 * cam_w), int(0.9 * cam_h)),
        (int(0.9 * cam_w), int(0.9 * cam_h)),
    ]
    for tx, ty in targets:
        cv2.circle(frame, (tx, ty), 8, (100, 100, 255), 1)
        cv2.line(frame, (tx - 12, ty), (tx + 12, ty), (100, 100, 255), 1)
        cv2.line(frame, (tx, ty - 12), (tx, ty + 12), (100, 100, 255), 1)
    return frame


def _init_nsapp():
    """NSApplication을 메인 스레드에서 초기화 (cv2.imshow 이전에 한 번만 호출)"""
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        app = NSApplication.sharedApplication()
        # 독(Dock) 아이콘 없이 액세서리 앱으로 실행
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except ImportError:
        pass
    except Exception as e:
        print(f"[Info] NSApplication 초기화 실패 (무시됨): {e}")


def main():
    parser = argparse.ArgumentParser(description="Step 1 — 눈 트래킹 타겟팅 테스트")
    parser.add_argument("--no-mouse", action="store_true", help="마우스 커서 이동 비활성화")
    parser.add_argument("--debug", action="store_true", help="랜드마크 시각화 활성화")
    args = parser.parse_args()

    # ── NSApplication 메인 스레드 초기화 (오버레이 / cv2 보다 먼저) ──
    _init_nsapp()

    # 설정 로드
    config = load_config()

    # 화면 크기
    screen_w, screen_h = get_screen_size()
    print(f"[Info] 화면 해상도: {screen_w}x{screen_h}")

    # 모듈 초기화
    cam_cfg = config.get("camera", {})
    camera = CameraCapture(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    )

    if not camera.start():
        print("[ERROR] 카메라 시작 실패. 종료합니다.")
        sys.exit(1)

    eye_tracker = EyeTracker(config)
    gaze_estimator = GazeEstimator(screen_w, screen_h, config)

    # 빨간 테두리 오버레이 시작 (메인 스레드에서 NSWindow 생성)
    border_overlay = ScreenBorderOverlayV2(border_width=5)
    border_overlay.start()

    # 호버 오버레이 + 감지기 시작 (메인 스레드에서 NSWindow 생성)
    hover_overlay = HoverOverlay(dwell_threshold=0.8)
    hover_overlay.start()
    hover_detector = HoverDetector(dwell_threshold=0.8)

    # 마우스 컨트롤러 (--no-mouse 아닐 때만)
    mouse = None
    mouse_enabled = not args.no_mouse
    if mouse_enabled:
        try:
            from action.mouse_controller import MouseController
            mouse = MouseController()
        except Exception as e:
            print(f"[WARN] 마우스 컨트롤러 초기화 실패: {e}")
            mouse_enabled = False

    # 눈 감기 3초 → 초점 재설정 감지기
    blink_detector = BlinkResetDetector()

    # 상태 변수
    show_debug = args.debug
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    eye_detected = False
    gaze_point = None

    print("\n" + "="*50)
    print("  Open Pilot — Step 1 눈 트래킹 타겟팅 테스트")
    print("="*50)
    print("  c   : 캘리브레이션 시작")
    print("  d   : 디버그 시각화 토글")
    print("  m   : 마우스 커서 이동 토글")
    print("  r   : 스무딩 리셋")
    print("  q   : 종료")
    print("="*50 + "\n")

    while True:
        frame = camera.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]

        # FPS 계산
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # 눈 추적
        eye_data = eye_tracker.process(frame)
        eye_detected = eye_data is not None

        # ── 눈 감기 3초 → 초점 재설정 ────────────────────────
        did_reset = blink_detector.update(eye_data)
        if did_reset:
            gaze_estimator.reset()
            print("[Info] 눈 깜빡임 감지 → 시선 초점 재설정")

        # 시선 추정
        if eye_data is not None and not blink_detector.is_waiting_reset:
            if gaze_estimator.is_calibrating:
                cal_status = gaze_estimator.update_calibration(eye_data)
                if cal_status["done"]:
                    print("[Info] 캘리브레이션 완료!")
            else:
                gaze_point = gaze_estimator.estimate(eye_data)

                if gaze_point:
                    # 마우스 커서 이동
                    if mouse_enabled and mouse:
                        mouse.move(gaze_point.x, gaze_point.y)

                    # 호버 감지 → 공유 상태 업데이트
                    hovered = hover_detector.update(gaze_point.x, gaze_point.y)
                    hover_state.set(hovered)

        # 호버 오버레이 갱신 (메인 스레드에서 매 프레임 호출)
        hover_overlay.refresh()

        # ─── 화면 그리기 ───────────────────────────────────

        # 디버그 시각화
        if show_debug and eye_data is not None:
            frame = eye_tracker.draw_debug(frame, eye_data)

        # 캘리브레이션 화면
        if gaze_estimator.is_calibrating:
            cal_pt = gaze_estimator.current_calibration_point
            if cal_pt:
                cal_status = gaze_estimator.update_calibration(eye_data) if eye_data else \
                    {"done": False, "current_idx": 0, "total": 5, "progress": 0.0}
                frame = draw_calibration_screen(
                    frame, cal_pt,
                    cal_status["progress"],
                    cal_status["current_idx"],
                    cal_status["total"],
                    cam_w, cam_h, screen_w, screen_h
                )
        else:
            # 타겟 그리드 (5개 기준점)
            frame = draw_target_grid(frame, cam_w, cam_h)

            # 시선 포인터 (짙은 회색 원형)
            if gaze_point is not None:
                frame = draw_gaze_pointer(
                    frame, gaze_point.x, gaze_point.y,
                    screen_w, screen_h, cam_w, cam_h
                )

        # 호버 배지 (카메라 프리뷰 하단)
        frame = draw_hover_badge(frame, hover_detector.current_element, cam_w, cam_h)

        # 눈 감기 진행 표시
        frame = draw_blink_indicator(frame, blink_detector, cam_w, cam_h)

        # HUD
        frame = draw_hud(frame, current_fps, gaze_estimator.is_calibrated,
                         mouse_enabled, eye_detected, blink_detector.close_progress)

        cv2.imshow("Open Pilot — Step 1 Eye Targeting Test", frame)

        # ─── 키 입력 처리 ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # q 또는 ESC
            break
        elif key == ord('c'):
            gaze_estimator.start_calibration()
        elif key == ord('r'):
            gaze_estimator.reset()
            print("[Info] 스무딩 상태 초기화")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[Info] 디버그 시각화: {'ON' if show_debug else 'OFF'}")
        elif key == ord('m'):
            mouse_enabled = not mouse_enabled
            if mouse_enabled and mouse is None:
                try:
                    from action.mouse_controller import MouseController
                    mouse = MouseController()
                except Exception as e:
                    print(f"[WARN] 마우스 컨트롤러 초기화 실패: {e}")
                    mouse_enabled = False
            print(f"[Info] 마우스 제어: {'ON' if mouse_enabled else 'OFF'}")

    # 종료
    hover_state.set(None)
    hover_overlay.stop()
    border_overlay.stop()
    camera.stop()
    eye_tracker.close()
    cv2.destroyAllWindows()
    print("\n[Info] 종료됨")


if __name__ == "__main__":
    main()
