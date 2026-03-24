"""
Step 4 — Unified Pilot Agent
=============================
눈 + 손 + 음성을 하나의 루프에서 동시 제어합니다.

  눈  → 마우스 커서 이동 (시선 추적)
  손  → 클릭 / 스크롤 / 줌 / 창 닫기
  음성 → 텍스트 입력 / AI 명령 (선택)

조작 키 (미리보기 창):
  d      — 디버그 뷰 전환 (Eye / Hand / Off)
  c      — 시선 캘리브레이션 시작
  r      — 시선 스무딩 리셋
  SPACE  — 음성 녹음 시작 / 중지 (토글)
  TAB    — 음성 모드 전환 (DICTATE ↔ AI CMD)
  m      — 마우스 이동 ON/OFF
  q/ESC  — 종료
"""
import cv2
import numpy as np
import subprocess
import time
import threading
import signal
import sys
import os
from typing import Optional, List

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from core.vision.camera_capture import CameraCapture
from core.vision.eye_tracker import EyeTracker
from core.vision.gaze_estimator import GazeEstimator
from core.vision.hand_tracker import HandTracker, HandGesture, HandData, PalmRubDetector
from core.vision.hand_tracker import THUMB_TIP, INDEX_TIP
from core.audio.audio_capture import AudioCapture
from core.audio.whisper_transcriber import WhisperTranscriber
from action.mouse_controller import MouseController
from action.click_controller import ClickController
from action.scroll_controller import ScrollController
from action.zoom_controller import ZoomController
from action import keyboard_controller as kbd
from action.system_controller import execute_system_command
from core.ai.local_command_parser import parse_voice_command
from core.ai.gemini_agent import GeminiAgent, is_available as gemini_available
from feedback.screen_overlay import ScreenBorderOverlayV2
from feedback.gesture_status_overlay import GestureStatusOverlay
from feedback.voice_overlay import VoiceOverlay

# 시선 커서 오버레이 (실제 화면)
try:
    from feedback.gaze_cursor_overlay import GazeCursorOverlay
    _HAS_GAZE_OVERLAY = True
except ImportError:
    _HAS_GAZE_OVERLAY = False

# AI 모드
try:
    from core.ai.command_processor import CommandProcessor
    _AI_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    _AI_AVAILABLE = False

# Gemini AI 에이전트
_gemini_ok = gemini_available()
if _gemini_ok:
    print("[GeminiAgent] ✨ Gemini API 감지 — SMART 음성 모드 사용 가능")

# ── 상수 ──────────────────────────────────────────────────────────
PREVIEW_W = 480
PREVIEW_H = 270

MODE_DICTATE = "DICTATE"
MODE_SMART   = "SMART"       # 로컬 매칭 → OpenClaw/Gemini 3 폴백

DEBUG_EYE  = "eye"
DEBUG_HAND = "hand"
DEBUG_OFF  = "off"


def _get_screen_size(config: dict):
    scr = config.get("screen", {})
    w, h = scr.get("width"), scr.get("height")
    if w and h:
        return int(w), int(h)
    try:
        from AppKit import NSScreen
        f = NSScreen.mainScreen().frame()
        return int(f.size.width), int(f.size.height)
    except Exception:
        return 1440, 900


# ── 미리보기 HUD ──────────────────────────────────────────────────

GESTURE_LABELS = {
    HandGesture.PALM:         "PALM",
    HandGesture.PINCH_INDEX:  "PINCH-L",
    HandGesture.PINCH_MIDDLE: "PINCH-R",
    HandGesture.FIST:         "FIST",
    HandGesture.NONE:         "--",
}
GESTURE_COLORS = {
    HandGesture.PALM:         (80,  220,  80),
    HandGesture.PINCH_INDEX:  (80,   80, 255),
    HandGesture.PINCH_MIDDLE: (80,   80, 255),
    HandGesture.FIST:         (0,   165, 255),
    HandGesture.NONE:         (80,   80,  80),
}
MODE_COLORS_CV = {
    MODE_DICTATE: (80, 220, 80),
    MODE_SMART:   (200, 80, 255),
}


def draw_hud(frame, eye_detected: bool, mouse_on: bool, is_calibrated: bool,
             left: Optional[HandData], right: Optional[HandData],
             zoom_active: bool, rub_active: bool, rub_progress: float,
             voice_mode: str, is_recording: bool,
             debug_mode: str, cam_w: int, cam_h: int) -> np.ndarray:

    cv2.rectangle(frame, (0, 0), (cam_w, 30), (18, 18, 18), -1)

    x = 6

    # 눈 상태
    eye_c = (80, 220, 80) if eye_detected else (80, 80, 220)
    eye_s = f"Eye:{'ON' if eye_detected else '--'}"
    cv2.putText(frame, eye_s, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, eye_c, 1)
    x += 72

    # 마우스
    m_c = (80, 220, 80) if mouse_on else (80, 80, 80)
    cv2.putText(frame, f"Mouse:{'ON' if mouse_on else 'OFF'}", (x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, m_c, 1)
    x += 90

    # 캘리브레이션
    cal_c = (80, 220, 80) if is_calibrated else (180, 120, 40)
    cv2.putText(frame, f"Cal:{'OK' if is_calibrated else 'No'}", (x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, cal_c, 1)
    x += 68

    # 왼손 제스처
    def hand_txt(hand, side):
        if hand is None:
            return f"{side}:--"
        g = hand.gesture
        return f"{side}:{GESTURE_LABELS.get(g,'?')}"

    lc = GESTURE_COLORS.get(left.gesture if left else HandGesture.NONE, (80, 80, 80))
    cv2.putText(frame, hand_txt(left, "L"), (x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, lc, 1)
    x += 88
    rc = GESTURE_COLORS.get(right.gesture if right else HandGesture.NONE, (80, 80, 80))
    cv2.putText(frame, hand_txt(right, "R"), (x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, rc, 1)
    x += 88

    # 음성 모드
    vc = MODE_COLORS_CV.get(voice_mode, (120, 120, 120))
    rec_s = " ●REC" if is_recording else ""
    cv2.putText(frame, f"{voice_mode}{rec_s}", (x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, vc, 1)

    # 조작 힌트 (하단)
    hint = "d=view  c=cal  r=reset  SPACE=voice  m=mouse  q=quit"
    cv2.putText(frame, hint, (6, cam_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (55, 55, 55), 1)

    # Rub 진행 바
    if rub_active and rub_progress > 0:
        bw = int(cam_w * 0.4)
        bf = int(bw * rub_progress)
        bx = (cam_w - bw) // 2
        by = cam_h - 22
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 5), (35, 35, 35), -1)
        cv2.rectangle(frame, (bx, by), (bx + bf, by + 5), (80, 200, 255), -1)
        cv2.putText(frame, "RUB TO CLOSE", (bx, by - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 200, 255), 1)

    return frame


# ── 음성 변환 워커 ────────────────────────────────────────────────

class _TranscribeWorker(threading.Thread):
    def __init__(self, audio, transcriber, mode, gemini_agent, event_log, on_done):
        super().__init__(daemon=True)
        self._audio, self._tr = audio, transcriber
        self._mode, self._gemini = mode, gemini_agent
        self._log, self._on_done = event_log, on_done

    def run(self):
        self._on_done("", "Transcribing...")
        try:
            text = self._tr.transcribe(self._audio)
        except Exception as e:
            self._on_done("", f"Whisper error: {e}")
            return
        if not text:
            self._on_done("", "Nothing heard.")
            return
        print(f"[Whisper] {text}")

        if self._mode == MODE_DICTATE:
            # ── DICTATE: 그대로 텍스트 입력 ──────────────────
            kbd.type_text(text)
            self._log.append(f"Typed: {text[:40]}")
            self._on_done(text, "Done.")

        elif self._mode == MODE_SMART:
            # ── SMART: 로컬 매칭 → Gemini 3 폴백 ────────────
            action = parse_voice_command(text)
            if action:
                # 1단계: 로컬 키워드 매칭 성공 → 즉시 실행
                print(f"[SMART] ⚡ 로컬 매칭: {action}")
                self._run_action(action)
                self._log.append(f"LOCAL: {action.get('action','?')}")
                self._on_done(text, f"⚡ {action.get('action','?')}")
            elif self._gemini:
                # 2단계: Gemini 3 AI 에이전트로 자연어 처리
                print(f"[SMART] 로컬 매칭 실패 → ✨ Gemini 3: {text}")
                self._on_done(text, "✨ Gemini 3 처리 중...")
                try:
                    action = self._gemini.process(text)
                    act_type = action.get("action", "say")
                    print(f"[SMART/Gemini3] 액션: {action}")

                    if act_type == "say":
                        # 응답만 표시 (실행 없음)
                        msg = action.get("text", "")[:60]
                        print(f"[Gemini3] 💬 {msg}")
                        self._log.append(f"✨ {msg}")
                        self._on_done(text, f"💬 {msg}")
                    elif act_type == "shell":
                        # 셸 명령 실행
                        cmd = action.get("command", "")
                        print(f"[Gemini3] 🖥 shell: {cmd}")
                        try:
                            r = subprocess.run(
                                cmd, shell=True, capture_output=True,
                                text=True, timeout=10
                            )
                            output = r.stdout.strip()[:60] or "Done"
                            self._log.append(f"✨ shell: {cmd[:30]}")
                            self._on_done(text, f"✨ {output}")
                        except Exception as e:
                            self._on_done(text, f"shell 실패: {e}")
                    else:
                        # open / shortcut / type / scroll / system
                        self._run_action(action)
                        self._log.append(f"✨ {act_type}")
                        self._on_done(text, f"✨ {act_type}")
                except Exception as e:
                    print(f"[SMART/Gemini3] 오류: {e}")
                    self._on_done(text, f"AI 오류: {e}")
            else:
                # Gemini 미설정 시 텍스트 입력으로 폴백
                print(f"[SMART] 매칭 실패 + Gemini 없음 → 텍스트 입력")
                kbd.type_text(text)
                self._log.append(f"Typed: {text[:40]}")
                self._on_done(text, "No match → typed.")
        else:
            kbd.type_text(text)
            self._log.append(f"Typed: {text[:40]}")
            self._on_done(text, "Done.")

    def _run_action(self, action):
        act = action.get("action", "")
        if act == "type":
            kbd.type_text(action.get("text", ""))
        elif act == "shortcut":
            kbd.press_keys(action.get("keys", []))
        elif act == "open":
            kbd.open_target(action.get("target", ""))
        elif act == "scroll":
            from action.scroll_controller import _do_scroll
            d = action.get("direction", "down")
            _do_scroll(80 * int(action.get("amount", 3)) * (1 if d == "up" else -1))
        elif act == "system":
            execute_system_command(action.get("command", ""))
        elif act == "say":
            pass  # 응답만 표시
        self._log.append(f"{act}: {list(action.values())[1] if len(action) > 1 else ''}")


# ── 손 스켈레톤 간이 표시 (디버그 OFF 시) ──────────────────────────

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # 엄지
    (0,5),(5,6),(6,7),(7,8),        # 검지
    (0,9),(9,10),(10,11),(11,12),   # 중지
    (0,13),(13,14),(14,15),(15,16), # 약지
    (0,17),(17,18),(18,19),(19,20), # 소지
    (5,9),(9,13),(13,17),           # 손바닥 가로
]


def _draw_hand_skeleton(frame: np.ndarray, hand) -> np.ndarray:
    """손 뼈대를 간단한 막대(stick)로 표시"""
    if hand is None or hand.landmarks is None:
        return frame
    h, w = frame.shape[:2]
    lms = hand.landmarks

    def px(lm):
        return int(lm.x * w), int(lm.y * h)

    # 색상: 왼손=초록, 오른손=파랑
    color = (80, 220, 80) if hand.handedness == "Left" else (255, 160, 80)
    joint_color = (255, 255, 255)

    # 뼈대 연결선
    for a, b in _HAND_CONNECTIONS:
        if a < len(lms) and b < len(lms):
            cv2.line(frame, px(lms[a]), px(lms[b]), color, 2, cv2.LINE_AA)

    # 관절 점
    for i, lm in enumerate(lms):
        p = px(lm)
        r = 5 if i in (4, 8, 12, 16, 20) else 3  # 손끝은 더 크게
        cv2.circle(frame, p, r, joint_color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, color, 1, cv2.LINE_AA)

    # 제스처 라벨
    g_label = GESTURE_LABELS.get(hand.gesture, "--")
    wrist = px(lms[0])
    cv2.putText(frame, f"{hand.handedness}:{g_label}",
                (wrist[0] - 10, wrist[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


# ── 메인 루프 ─────────────────────────────────────────────────────

def run(model_size: str = "base", voice_enabled: bool = True):
    config = load_config()
    screen_w, screen_h = _get_screen_size(config)
    print(f"[Unified] 화면 해상도: {screen_w}x{screen_h}")

    # ── Ctrl+C 시그널 핸들링 ──────────────────────────────────────
    _shutdown = threading.Event()

    def _sigint_handler(sig, frame_):
        print("\n[Unified] Ctrl+C 감지 — 종료합니다...")
        _shutdown.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── NSApplication 초기화 ─────────────────────────────────────
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )
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

    # ── 눈 추적 ──────────────────────────────────────────────────
    eye_tracker    = EyeTracker(config)
    gaze_estimator = GazeEstimator(screen_w, screen_h, config)

    # ── 손 추적 ──────────────────────────────────────────────────
    hand_tracker = HandTracker(config)
    click_ctrl   = ClickController(cooldown_ms=config.get("hand_tracking", {}).get("click_cooldown_ms", 600))
    scroll_ctrl  = ScrollController()
    zoom_ctrl    = ZoomController()
    rub_detector = PalmRubDetector()

    # ── 마우스 ──────────────────────────────────────────────────
    mouse = None
    mouse_on = True
    try:
        mouse = MouseController()
    except Exception as e:
        print(f"[WARN] 마우스 컨트롤러 초기화 실패: {e}")
        mouse_on = False

    # ── 음성 ─────────────────────────────────────────────────────
    audio_cap    = None
    transcriber  = None
    gemini_agent = None
    if voice_enabled:
        try:
            audio_cap   = AudioCapture()
            transcriber = WhisperTranscriber(model_size=model_size)
        except Exception as e:
            print(f"[WARN] 음성 모듈 초기화 실패: {e}")
            voice_enabled = False

    # ── Gemini AI 에이전트 ─────────────────────────────────────
    if _gemini_ok:
        try:
            gemini_agent = GeminiAgent()
            print("[GeminiAgent] ✨ Gemini 3 에이전트 준비 완료")
        except Exception as e:
            print(f"[WARN] Gemini 초기화 실패: {e}")
            gemini_agent = None

    voice_status  = ""
    transcription = ""
    event_log: list = []
    voice_busy    = False

    def on_transcribe_done(text, status):
        nonlocal transcription, voice_status, voice_busy
        transcription = text
        voice_status  = status
        voice_busy    = False

    # ── 오버레이 (메인 스레드에서 생성) ──────────────────────────
    border_overlay  = ScreenBorderOverlayV2(border_width=5, mode="real")
    border_overlay.start()

    gesture_overlay = GestureStatusOverlay()
    gesture_overlay.start()

    voice_overlay = VoiceOverlay()
    voice_overlay.start()

    gaze_cursor = None
    if _HAS_GAZE_OVERLAY:
        try:
            gaze_cursor = GazeCursorOverlay()
            gaze_cursor.start()
        except Exception:
            pass

    # ── 상태 ────────────────────────────────────────────────────
    debug_mode  = DEBUG_EYE
    eye_detected = False
    gaze_point  = None
    prev_hand: dict = {"Left": HandGesture.NONE, "Right": HandGesture.NONE}
    zoom_active = False
    zoom_delta  = 0.0

    # 기본 음성 모드: Gemini 있으면 SMART, 없으면 DICTATE
    voice_mode = MODE_SMART if gemini_agent else MODE_DICTATE

    print("\n" + "=" * 60)
    print("  OpenPilot — Step 4  Unified Pilot Agent")
    print("=" * 60)
    print("  👁  Eye    → Mouse cursor movement")
    print("  ✋ Hand   → Click / Scroll / Zoom / Close window")
    print("  🎙  Voice  → SMART voice control  (SPACE)")
    print()
    print("  Voice Modes (TAB to switch):")
    print("    DICTATE — 말한 내용 그대로 텍스트 입력")
    gemini_s = "✅" if gemini_agent else "❌ (python main.py --setup)"
    print(f"    SMART   — 로컬 매칭 + ✨ Gemini 3 AI 폴백 {gemini_s}")
    print()
    print("  SMART 파이프라인:")
    print("    음성 → Whisper → 로컬 키워드 매칭 (50+ 명령, 즉시 실행)")
    print("                      └─ 실패 시 → ✨ Gemini 3 (자연어 이해)")
    print()
    print("  로컬 명령 예시: 크롬 열어 / 볼륨 업 / 스크린샷 / 닫기")
    print("  AI 명령 예시:   오늘 날씨 / 구글 검색 / 이메일 보내줘")
    print()
    print("  d        → Toggle debug view (Eye / Hand / Off)")
    print("  c        → Eye calibration")
    print("  r        → Reset gaze smoothing")
    print("  SPACE    → Voice recording toggle")
    print("  TAB      → Switch voice mode (SMART ↔ DICTATE)")
    print("  m        → Toggle mouse control")
    print("  Ctrl+C   → Quit")
    print("  q / ESC  → Quit")
    print("=" * 60 + "\n")

    while not _shutdown.is_set():
        frame = camera.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]

        # ══ 1. 눈 추적 → 커서 이동 ═══════════════════════════════
        eye_data = eye_tracker.process(frame)
        eye_detected = eye_data is not None

        if eye_data is not None:
            if gaze_estimator.is_calibrating:
                gaze_estimator.update_calibration(eye_data)
            else:
                gaze_point = gaze_estimator.estimate(eye_data)
                if gaze_point:
                    if mouse_on and mouse:
                        mouse.move(gaze_point.x, gaze_point.y)
                    if gaze_cursor:
                        try:
                            gaze_cursor.update_position(gaze_point.x, gaze_point.y)
                        except Exception:
                            pass

        # ══ 2. 손 추적 → 클릭 / 스크롤 / 줌 ═════════════════════
        hands: List[HandData] = hand_tracker.process_multi(frame)
        by_side = {h.handedness: h for h in hands}
        left  = by_side.get("Left")
        right = by_side.get("Right")

        curr_g = {
            "Left":  left.gesture  if left  else HandGesture.NONE,
            "Right": right.gesture if right else HandGesture.NONE,
        }

        # 양손 핀치 줌
        both_pinching = (
            curr_g["Left"]  == HandGesture.PINCH_INDEX and
            curr_g["Right"] == HandGesture.PINCH_INDEX
        )
        zoom_delta = 0.0

        if both_pinching:
            if not zoom_active:
                zoom_ctrl.start(left, right)
                zoom_active = True
                gesture_overlay.show("ZOOM START", "zoom")
            else:
                zoom_delta = zoom_ctrl.update(left, right)
                if abs(zoom_delta) > ZoomController.MIN_DELTA:
                    gesture_overlay.show(
                        "ZOOM IN" if zoom_delta > 0 else "ZOOM OUT", "zoom"
                    )
        else:
            if zoom_active:
                zoom_ctrl.stop()
                zoom_active = False

        # Palm rub → 창 닫기
        if not zoom_active:
            if rub_detector.update(left, right):
                kbd.press_keys(["cmd", "w"])
                gesture_overlay.show("CLOSE WINDOW", "click", duration=1.0)
                print("[Unified] Palm rub → Cmd+W")

        # 단일 손 제스처 (줌/rub 중엔 비활성)
        if not zoom_active and not rub_detector.is_active:
            for side, hand in [("Left", left), ("Right", right)]:
                cg = curr_g[side]
                pg = prev_hand[side]

                if cg != pg:
                    if pg == HandGesture.FIST:
                        scroll_ctrl.fist_release()

                    if cg == HandGesture.PINCH_INDEX:
                        if click_ctrl.left_click():
                            gesture_overlay.show("LEFT CLICK", "click")
                            print(f"[Unified] Left click ({side})")

                    elif cg == HandGesture.PINCH_MIDDLE:
                        if click_ctrl.right_click():
                            gesture_overlay.show("RIGHT CLICK", "click")
                            print(f"[Unified] Right click ({side})")

                if cg == HandGesture.FIST and hand:
                    scroll_ctrl.fist_update(hand.hand_center[1])
                    if scroll_ctrl._vel_buf:
                        vel = scroll_ctrl._vel_buf[-1]
                        if abs(vel) > 0.05:
                            gesture_overlay.show(
                                "SCROLL UP" if vel > 0 else "SCROLL DOWN",
                                "scroll", duration=0.35
                            )

        prev_hand.update(curr_g)

        # ══ 3. 음성 오버레이 갱신 ════════════════════════════════
        if audio_cap:
            voice_overlay.update(
                recording=audio_cap.is_recording,
                mode=voice_mode,
                transcription=transcription,
                status=voice_status,
                mic_level=audio_cap.level,
            )
        gesture_overlay.refresh()
        voice_overlay.refresh()

        # ══ 4. 미리보기 창 렌더링 ════════════════════════════════
        if debug_mode == DEBUG_EYE and eye_data is not None:
            gx = gaze_point.x if gaze_point else -1
            gy = gaze_point.y if gaze_point else -1
            debug_frame = eye_tracker.draw_debug(
                frame.copy(), eye_data,
                gaze_x=gx, gaze_y=gy,
                screen_w=screen_w, screen_h=screen_h,
            )
            preview = cv2.resize(debug_frame, (PREVIEW_W, PREVIEW_H))
        elif debug_mode == DEBUG_HAND:
            debug_frame = frame.copy()
            for h in hands:
                debug_frame = hand_tracker.draw_debug(debug_frame, h)
            if zoom_active and left and right:
                # 줌 연결선
                def tip_px(hd, fw, fh):
                    lms = hd.landmarks
                    mx = (lms[THUMB_TIP].x + lms[INDEX_TIP].x) / 2
                    my = (lms[THUMB_TIP].y + lms[INDEX_TIP].y) / 2
                    return int(mx * fw), int(my * fh)
                lp = tip_px(left,  cam_w, cam_h)
                rp = tip_px(right, cam_w, cam_h)
                c  = (0, 255, 180) if zoom_delta > 0 else (0, 120, 255)
                cv2.line(debug_frame, lp, rp, c, 2, cv2.LINE_AA)
            preview = cv2.resize(debug_frame, (PREVIEW_W, PREVIEW_H))
        else:
            # 디버그 OFF 시에도 손 스켈레톤 기본 표시
            base_frame = frame.copy()
            for h in hands:
                base_frame = _draw_hand_skeleton(base_frame, h)
            preview = cv2.resize(base_frame, (PREVIEW_W, PREVIEW_H))

        preview = draw_hud(
            preview, eye_detected, mouse_on, gaze_estimator.is_calibrated,
            left, right, zoom_active,
            rub_detector.is_active, rub_detector.progress,
            voice_mode,
            audio_cap.is_recording if audio_cap else False,
            debug_mode, PREVIEW_W, PREVIEW_H,
        )

        cv2.imshow("OpenPilot — Unified  [q=quit]", preview)
        key = cv2.waitKey(1) & 0xFF

        # ══ 5. 키 입력 ═══════════════════════════════════════════
        if key in (ord('q'), 27):
            break

        elif key == ord('d'):
            debug_mode = {DEBUG_EYE: DEBUG_HAND,
                          DEBUG_HAND: DEBUG_OFF,
                          DEBUG_OFF: DEBUG_EYE}[debug_mode]
            print(f"[Unified] Debug: {debug_mode}")

        elif key == ord('c'):
            gaze_estimator.start_calibration()
            print("[Unified] 캘리브레이션 시작")

        elif key == ord('r'):
            gaze_estimator.reset()
            print("[Unified] 시선 스무딩 리셋")

        elif key == ord('m'):
            mouse_on = not mouse_on
            print(f"[Unified] 마우스: {'ON' if mouse_on else 'OFF'}")

        elif key in (ord(' '), ord('r')) and audio_cap:
            if voice_busy:
                pass
            elif not audio_cap.is_recording:
                audio_cap.start_recording()
                voice_status  = "Recording..."
                transcription = ""
            else:
                audio = audio_cap.stop_recording()
                if len(audio) < 3200:
                    voice_status = "Too short."
                else:
                    voice_busy = True
                    _TranscribeWorker(
                        audio, transcriber, voice_mode, gemini_agent,
                        event_log, on_transcribe_done,
                    ).start()

        elif key == 9 and audio_cap:   # TAB
            if not audio_cap.is_recording and not voice_busy:
                # 2모드 토글: SMART ↔ DICTATE
                if voice_mode == MODE_SMART:
                    voice_mode   = MODE_DICTATE
                    voice_status = "DICTATE mode. (텍스트 입력)"
                else:
                    voice_mode   = MODE_SMART
                    voice_status = "SMART mode. (로컬 + 🦞 Gemini 3)"
                print(f"[Unified] 음성 모드: {voice_mode}")

    # ── 종료 ─────────────────────────────────────────────────────
    scroll_ctrl.stop_inertia()
    zoom_ctrl.stop()
    if audio_cap and audio_cap.is_recording:
        audio_cap.stop_recording()
    if gaze_cursor:
        try:
            gaze_cursor.stop()
        except Exception:
            pass
    gesture_overlay.stop()
    voice_overlay.stop()
    border_overlay.stop()
    camera.stop()
    eye_tracker.close()
    hand_tracker.close()
    cv2.destroyAllWindows()
    print("[Unified] 종료됨")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="base",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--no-voice", action="store_true")
    args = parser.parse_args()
    run(model_size=args.model, voice_enabled=not args.no_voice)
