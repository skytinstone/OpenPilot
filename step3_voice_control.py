"""
Step 3 - Voice Control (Whisper STT + Claude AI)
=================================================
두 가지 모드:

  [DICTATE]  — 음성 → Whisper → 텍스트 그대로 입력
  [AI CMD]   — 음성 → Whisper → Claude 해석 → 명령 실행

조작 키:
  SPACE / r  — 녹음 시작 / 중지 (토글)
  TAB        — DICTATE ↔ AI CMD 모드 전환
  q / ESC    — 종료
"""
import cv2
import numpy as np
import time
import threading
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from core.vision.camera_capture import CameraCapture
from core.audio.audio_capture import AudioCapture
from core.audio.whisper_transcriber import WhisperTranscriber
from action import keyboard_controller as kbd

# AI 모드는 ANTHROPIC_API_KEY 있을 때만 활성화
try:
    from core.ai.command_processor import CommandProcessor
    _AI_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    _AI_AVAILABLE = False


# ── 모드 상수 ────────────────────────────────────────────────────
MODE_DICTATE  = "DICTATE"
MODE_AI_CMD   = "AI CMD"

MODE_COLORS = {
    MODE_DICTATE: (80,  220, 80),
    MODE_AI_CMD:  (0,   180, 255),
}


# ── 액션 실행기 ──────────────────────────────────────────────────

def execute_action(action: dict, event_log: list):
    """Claude 가 반환한 액션 dict 실행"""
    act = action.get("action", "")

    if act == "type":
        text = action.get("text", "")
        kbd.type_text(text)
        event_log.append(f"Typed: {text[:40]}")

    elif act == "shortcut":
        keys = action.get("keys", [])
        kbd.press_keys(keys)
        event_log.append(f"Shortcut: {'+'.join(keys)}")

    elif act == "scroll":
        direction = action.get("direction", "down")
        amount    = int(action.get("amount", 3))
        from action.scroll_controller import _do_scroll
        px = 80 * amount * (1 if direction == "up" else -1)
        _do_scroll(px)
        event_log.append(f"Scroll {direction} x{amount}")

    elif act == "click":
        from action.click_controller import ClickController
        cc = ClickController()
        if action.get("button") == "right":
            cc.right_click()
            event_log.append("Right click")
        else:
            cc.left_click()
            event_log.append("Left click")

    elif act == "open":
        target = action.get("target", "")
        kbd.open_target(target)
        event_log.append(f"Open: {target}")

    elif act == "say":
        resp = action.get("text", "")
        event_log.append(f"AI: {resp}")

    else:
        event_log.append(f"Unknown action: {act}")


# ── 시각화 ───────────────────────────────────────────────────────

def draw_ui(frame, mode: str, is_recording: bool, mic_level: float,
            rec_seconds: float, status_text: str,
            transcription: str, last_events: list,
            cam_w: int, cam_h: int) -> np.ndarray:

    color = MODE_COLORS.get(mode, (120, 120, 120))

    # ── 상단 모드 바 ─────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (cam_w, 38), (18, 18, 18), -1)

    # 모드 이름
    cv2.putText(frame, f"[ {mode} ]", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 녹음 시간
    if is_recording:
        rec_str = f"REC  {rec_seconds:.1f}s"
        cv2.putText(frame, rec_str, (cam_w // 2 - 40, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)

    # 조작 힌트
    cv2.putText(frame, "SPACE=rec  TAB=mode  q=quit",
                (cam_w - 270, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1)

    # ── 녹음 표시등 ──────────────────────────────────────────────
    if is_recording:
        # 깜빡이는 빨간 원
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (cam_w - 22, 60), 10, (40, 40, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cam_w - 22, 60), 10, (100, 100, 255), 1, cv2.LINE_AA)

        # 마이크 음량 게이지 (우측 세로 바)
        bar_x  = cam_w - 14
        bar_h  = 180
        bar_y  = 80
        filled = int(bar_h * mic_level)
        cv2.rectangle(frame, (bar_x - 6, bar_y), (bar_x, bar_y + bar_h),
                      (40, 40, 40), -1)
        if filled > 0:
            bar_color = (60, 60, 255) if mic_level > 0.7 else (80, 180, 80)
            cv2.rectangle(frame, (bar_x - 6, bar_y + bar_h - filled),
                          (bar_x, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x - 6, bar_y), (bar_x, bar_y + bar_h),
                      (80, 80, 80), 1)

    # ── 마이크 아이콘 (중앙) ─────────────────────────────────────
    cx, cy = cam_w // 2, cam_h // 2 - 30
    mic_color = (60, 60, 255) if is_recording else (80, 80, 80)
    cv2.rectangle(frame, (cx - 14, cy - 30), (cx + 14, cy + 10), mic_color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy - 30), (14, 14), 0, 180, 360, mic_color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy - 30), (14, 14), 0, 0, 180, (50, 50, 50), -1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 10), (cx, cy + 26), mic_color, 3, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy + 10), (22, 16), 0, 0, 180, mic_color, 2, cv2.LINE_AA)

    # ── 상태 텍스트 ──────────────────────────────────────────────
    if status_text:
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, status_text, ((cam_w - tw) // 2, cy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── 변환된 텍스트 (하단 패널) ────────────────────────────────
    panel_y = cam_h - 130
    cv2.rectangle(frame, (0, panel_y), (cam_w, cam_h), (15, 15, 15), -1)
    cv2.line(frame, (0, panel_y), (cam_w, panel_y), (50, 50, 50), 1)

    # 마지막 변환 텍스트
    if transcription:
        # 길면 자르기
        max_chars = 60
        display = transcription if len(transcription) <= max_chars \
                  else "..." + transcription[-(max_chars - 3):]
        cv2.putText(frame, f'"{display}"',
                    (12, panel_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # 이벤트 로그 (최근 3개)
    for i, ev in enumerate(last_events[-3:]):
        cv2.putText(frame, f"  > {ev}", (12, panel_y + 50 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    return frame


# ── 변환 + 실행 스레드 ───────────────────────────────────────────

class TranscribeWorker(threading.Thread):
    """녹음 완료 후 백그라운드에서 Whisper → (AI) → 실행"""

    def __init__(self, audio: np.ndarray, transcriber: WhisperTranscriber,
                 mode: str, ai_processor, event_log: list,
                 on_done):
        super().__init__(daemon=True)
        self._audio       = audio
        self._transcriber = transcriber
        self._mode        = mode
        self._ai          = ai_processor
        self._event_log   = event_log
        self._on_done     = on_done   # callback(transcription, status)

    def run(self):
        self._on_done("", "Transcribing...")
        try:
            text = self._transcriber.transcribe(self._audio)
        except Exception as e:
            self._on_done("", f"Whisper error: {e}")
            return

        if not text:
            self._on_done("", "Nothing heard. Try again.")
            return

        print(f"[Whisper] {text}")

        if self._mode == MODE_DICTATE:
            kbd.type_text(text)
            self._event_log.append(f"Typed: {text[:50]}")
            self._on_done(text, "Done.")

        elif self._mode == MODE_AI_CMD and self._ai:
            self._on_done(text, "AI processing...")
            try:
                action = self._ai.process(text)
                print(f"[AI] action: {action}")
                execute_action(action, self._event_log)
                self._on_done(text, f"Done: {action.get('action','?')}")
            except Exception as e:
                self._on_done(text, f"AI error: {e}")
        else:
            # AI 사용 불가 → 그냥 타이핑
            kbd.type_text(text)
            self._event_log.append(f"Typed (no AI): {text[:50]}")
            self._on_done(text, "Done (dictate fallback).")


# ── 메인 루프 ────────────────────────────────────────────────────

def run(model_size: str = "base"):
    config  = load_config()
    cam_cfg = config.get("camera", {})

    # 카메라
    camera = CameraCapture(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    )
    if not camera.start():
        print("[ERROR] Camera failed to start")
        sys.exit(1)

    # 오디오 + Whisper
    audio_cap    = AudioCapture()
    transcriber  = WhisperTranscriber(model_size=model_size)

    # AI (선택사항)
    ai_processor = None
    if _AI_AVAILABLE:
        try:
            ai_processor = CommandProcessor()
            print("[AI] Claude 연결 완료")
        except Exception as e:
            print(f"[AI] 연결 실패: {e}")

    # 상태
    mode         = MODE_DICTATE
    status_text  = "Press SPACE to speak"
    transcription= ""
    event_log: list = []
    is_busy      = False   # Whisper/AI 처리 중

    def on_transcribe_done(text, status):
        nonlocal transcription, status_text, is_busy
        transcription = text
        status_text   = status
        is_busy       = False

    print("\n" + "=" * 54)
    print("  Open Pilot - Step 3  Voice Control")
    print("=" * 54)
    print(f"  Whisper model : {model_size}")
    print(f"  AI (Claude)   : {'Available' if ai_processor else 'Not available (set ANTHROPIC_API_KEY)'}")
    print()
    print("  SPACE / r  -> Start / Stop recording")
    print("  TAB        -> Switch DICTATE <-> AI CMD mode")
    print("  q / ESC    -> Quit")
    print("=" * 54 + "\n")

    while True:
        frame = camera.read()
        if frame is None:
            continue

        cam_h, cam_w = frame.shape[:2]

        # UI 그리기
        frame = draw_ui(
            frame, mode,
            audio_cap.is_recording, audio_cap.level,
            audio_cap.recorded_seconds,
            status_text, transcription, event_log,
            cam_w, cam_h,
        )

        cv2.imshow("Open Pilot - Step 3  Voice Control", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── 키 입력 ───────────────────────────────────────────
        if key in (ord('q'), 27):          # 종료
            break

        elif key in (ord(' '), ord('r')):  # 녹음 토글
            if is_busy:
                pass   # 처리 중엔 무시
            elif not audio_cap.is_recording:
                audio_cap.start_recording()
                status_text = "Recording... (SPACE to stop)"
                transcription = ""
            else:
                audio = audio_cap.stop_recording()
                if len(audio) < 3200:
                    status_text = "Too short. Try again."
                else:
                    is_busy = True
                    worker = TranscribeWorker(
                        audio, transcriber, mode, ai_processor,
                        event_log, on_transcribe_done,
                    )
                    worker.start()

        elif key == 9:                     # TAB — 모드 전환
            if not audio_cap.is_recording and not is_busy:
                if mode == MODE_DICTATE:
                    if ai_processor:
                        mode = MODE_AI_CMD
                        status_text = "AI CMD mode. Press SPACE to speak."
                    else:
                        status_text = "AI unavailable. Set ANTHROPIC_API_KEY."
                else:
                    mode = MODE_DICTATE
                    status_text = "DICTATE mode. Press SPACE to speak."

    # ── 종료 ─────────────────────────────────────────────────────
    if audio_cap.is_recording:
        audio_cap.stop_recording()
    camera.stop()
    cv2.destroyAllWindows()
    print("[Step3] Stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size")
    args = parser.parse_args()
    run(model_size=args.model)
