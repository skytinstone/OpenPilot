"""
Step 3 — Real Screen Voice Control (Whisper + Claude AI)
=========================================================
음성 인식 결과를 실제 화면에 오버레이로 표시합니다.

  - 작은 카메라 창 (480×270): 마이크 UI + 키 입력 전용
  - 실제 화면 우측 상단: 모드 배지 (DICTATE / AI CMD) + REC 표시
  - 실제 화면 하단: 인식된 텍스트 토스트 (3초 후 fade-out)

두 가지 모드:
  [DICTATE]  — 음성 → Whisper → 텍스트 그대로 입력
  [AI CMD]   — 음성 → Whisper → Claude 해석 → 명령 실행

조작 키 (카메라 창에서):
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
from feedback.screen_overlay import ScreenBorderOverlayV2
from feedback.voice_overlay import VoiceOverlay

# AI 모드: ANTHROPIC_API_KEY 있을 때만 활성화
try:
    from core.ai.command_processor import CommandProcessor
    _AI_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    _AI_AVAILABLE = False

# 카메라 미리보기 크기
PREVIEW_W = 480
PREVIEW_H = 270

# 모드
MODE_DICTATE = "DICTATE"
MODE_AI_CMD  = "AI CMD"

MODE_COLORS = {
    MODE_DICTATE: (80,  220, 80),
    MODE_AI_CMD:  (0,   180, 255),
}


# ── 액션 실행기 (step3_voice_control 과 동일) ──────────────────────

def _execute_action(action: dict, event_log: list):
    act = action.get("action", "")
    if act == "type":
        kbd.type_text(action.get("text", ""))
        event_log.append(f"Typed: {action.get('text','')[:40]}")
    elif act == "shortcut":
        keys = action.get("keys", [])
        kbd.press_keys(keys)
        event_log.append(f"Shortcut: {'+'.join(keys)}")
    elif act == "scroll":
        direction = action.get("direction", "down")
        amount    = int(action.get("amount", 3))
        from action.scroll_controller import _do_scroll
        _do_scroll(80 * amount * (1 if direction == "up" else -1))
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
        kbd.open_target(action.get("target", ""))
        event_log.append(f"Open: {action.get('target','')}")
    elif act == "say":
        event_log.append(f"AI: {action.get('text','')}")
    else:
        event_log.append(f"Unknown: {act}")


# ── 미리보기 창 UI ────────────────────────────────────────────────

def draw_preview(frame, mode: str, is_recording: bool, mic_level: float,
                 rec_seconds: float, status_text: str,
                 last_events: list, cam_w: int, cam_h: int) -> np.ndarray:
    color = MODE_COLORS.get(mode, (120, 120, 120))

    # 상단 바
    cv2.rectangle(frame, (0, 0), (cam_w, 32), (18, 18, 18), -1)
    cv2.putText(frame, f"[ {mode} ]", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if is_recording:
        blink = int(time.time() * 2) % 2 == 0
        rec_str = f"REC  {rec_seconds:.1f}s"
        rec_color = (60, 60, 255) if blink else (100, 100, 180)
        cv2.putText(frame, rec_str, (cam_w // 2 - 36, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_color, 2)

        # 마이크 레벨 바 (우측)
        bar_x  = cam_w - 12
        bar_h  = 100
        bar_y  = 40
        filled = int(bar_h * min(mic_level, 1.0))
        cv2.rectangle(frame, (bar_x - 6, bar_y),
                      (bar_x, bar_y + bar_h), (40, 40, 40), -1)
        if filled > 0:
            lv_color = (60, 60, 255) if mic_level > 0.7 else (60, 180, 60)
            cv2.rectangle(frame,
                          (bar_x - 6, bar_y + bar_h - filled),
                          (bar_x, bar_y + bar_h), lv_color, -1)

    # 마이크 아이콘 (미리보기 중앙)
    cx, cy = cam_w // 2, cam_h // 2 - 20
    mic_c  = (60, 60, 255) if is_recording else (70, 70, 70)
    cv2.rectangle(frame, (cx - 12, cy - 26), (cx + 12, cy + 8), mic_c, -1)
    cv2.ellipse(frame, (cx, cy - 26), (12, 12), 0, 180, 360, mic_c, -1)
    cv2.ellipse(frame, (cx, cy - 26), (12, 12), 0,   0, 180, (40, 40, 40), -1)
    cv2.line(frame, (cx, cy + 8), (cx, cy + 22), mic_c, 3)
    cv2.ellipse(frame, (cx, cy + 8), (19, 13), 0, 0, 180, mic_c, 2)

    # 상태 텍스트
    if status_text:
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, status_text, ((cam_w - tw) // 2, cy + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 이벤트 로그 (하단 패널)
    panel_y = cam_h - 70
    cv2.rectangle(frame, (0, panel_y), (cam_w, cam_h), (14, 14, 14), -1)
    cv2.line(frame, (0, panel_y), (cam_w, panel_y), (45, 45, 45), 1)
    for i, ev in enumerate(last_events[-2:]):
        cv2.putText(frame, f"> {ev}", (8, panel_y + 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # 조작 힌트
    cv2.putText(frame, "SPACE=rec  TAB=mode  q=quit",
                (8, cam_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (55, 55, 55), 1)
    return frame


# ── 변환 워커 ─────────────────────────────────────────────────────

class _TranscribeWorker(threading.Thread):
    def __init__(self, audio, transcriber, mode, ai_processor,
                 event_log, on_done):
        super().__init__(daemon=True)
        self._audio       = audio
        self._transcriber = transcriber
        self._mode        = mode
        self._ai          = ai_processor
        self._event_log   = event_log
        self._on_done     = on_done

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
                print(f"[AI] {action}")
                _execute_action(action, self._event_log)
                self._on_done(text, f"Done: {action.get('action','?')}")
            except Exception as e:
                self._on_done(text, f"AI error: {e}")
        else:
            kbd.type_text(text)
            self._event_log.append(f"Typed (no AI): {text[:50]}")
            self._on_done(text, "Done (dictate fallback).")


# ── 메인 루프 ─────────────────────────────────────────────────────

def run(model_size: str = "base"):
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

    # ── 오디오 + Whisper ─────────────────────────────────────────
    audio_cap   = AudioCapture()
    transcriber = WhisperTranscriber(model_size=model_size)

    # ── AI (선택사항) ─────────────────────────────────────────────
    ai_processor = None
    if _AI_AVAILABLE:
        try:
            ai_processor = CommandProcessor()
            print("[AI] Claude 연결 완료")
        except Exception as e:
            print(f"[AI] 연결 실패: {e}")

    # ── 오버레이 (메인 스레드에서 생성) ──────────────────────────
    border_overlay = ScreenBorderOverlayV2(border_width=5)
    border_overlay.start()

    voice_overlay = VoiceOverlay()
    voice_overlay.start()

    # ── 상태 ────────────────────────────────────────────────────
    mode          = MODE_DICTATE
    status_text   = "Press SPACE to speak"
    transcription = ""
    event_log: list = []
    is_busy       = False

    def on_done(text, status):
        nonlocal transcription, status_text, is_busy
        transcription = text
        status_text   = status
        is_busy       = False

    print("\n" + "=" * 56)
    print("  OpenPilot — Step 3  Voice Control  (Real Screen)")
    print("=" * 56)
    print(f"  Whisper model : {model_size}")
    print(f"  AI (Claude)   : {'Available' if ai_processor else 'Not available (set ANTHROPIC_API_KEY)'}")
    print()
    print("  SPACE / r  -> Start / Stop recording")
    print("  TAB        -> Switch DICTATE <-> AI CMD mode")
    print("  q / ESC    -> Quit")
    print("=" * 56)
    print()
    print("  ★ Recording state & transcription shown on your actual screen.")
    print()

    while True:
        frame = camera.read()
        if frame is None:
            continue

        # ── 오버레이 상태 업데이트 ────────────────────────────────
        voice_overlay.update(
            recording=audio_cap.is_recording,
            mode=mode,
            transcription=transcription,
            status=status_text,
            mic_level=audio_cap.level,
        )
        voice_overlay.refresh()

        # ── 미리보기 창 렌더링 ────────────────────────────────────
        preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        preview = draw_preview(
            preview, mode,
            audio_cap.is_recording, audio_cap.level,
            audio_cap.recorded_seconds,
            status_text, event_log,
            PREVIEW_W, PREVIEW_H,
        )

        cv2.imshow("OpenPilot — Voice Control  [q=quit]", preview)
        key = cv2.waitKey(1) & 0xFF

        # ── 키 입력 ───────────────────────────────────────────────
        if key in (ord('q'), 27):
            break

        elif key in (ord(' '), ord('r')):
            if is_busy:
                pass
            elif not audio_cap.is_recording:
                audio_cap.start_recording()
                status_text   = "Recording... (SPACE to stop)"
                transcription = ""
            else:
                audio = audio_cap.stop_recording()
                if len(audio) < 3200:
                    status_text = "Too short. Try again."
                else:
                    is_busy = True
                    _TranscribeWorker(
                        audio, transcriber, mode, ai_processor,
                        event_log, on_done,
                    ).start()

        elif key == 9:   # TAB
            if not audio_cap.is_recording and not is_busy:
                if mode == MODE_DICTATE:
                    if ai_processor:
                        mode        = MODE_AI_CMD
                        status_text = "AI CMD mode. Press SPACE to speak."
                    else:
                        status_text = "AI unavailable. Set ANTHROPIC_API_KEY."
                else:
                    mode        = MODE_DICTATE
                    status_text = "DICTATE mode. Press SPACE to speak."

    # ── 종료 ─────────────────────────────────────────────────────
    if audio_cap.is_recording:
        audio_cap.stop_recording()
    voice_overlay.stop()
    border_overlay.stop()
    camera.stop()
    cv2.destroyAllWindows()
    print("[Step3 Real] 종료됨")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium"])
    args = parser.parse_args()
    run(model_size=args.model)
