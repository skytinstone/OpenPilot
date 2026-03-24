#!/usr/bin/env python3
"""
Linux 음성 명령 테스트 스크립트 (단독 실행)

필수 패키지:
  pip install faster-whisper sounddevice soundfile numpy

Linux 시스템 도구 (명령 실행용):
  sudo apt install xdotool xclip pulseaudio-utils  # Ubuntu/Debian
  sudo dnf install xdotool xclip pulseaudio-utils   # Fedora

실행:
  python test_voice_linux.py
  python test_voice_linux.py --lang ko       # 한국어 고정
  python test_voice_linux.py --lang en       # 영어 고정
  python test_voice_linux.py --model small   # 더 정확한 모델
  python test_voice_linux.py --no-exec       # 인식만, 실행 안 함
"""
import argparse
import re
import signal
import subprocess
import sys
import time
import threading
from typing import Optional

import numpy as np

# ── 오디오 녹음 ────────────────────────────────────────────────

RATE = 16000
CHANNELS = 1
CHUNK = 1024


class MicRecorder:
    def __init__(self):
        import sounddevice  # noqa: F401 — 설치 확인
        self._frames = []
        self._stream = None

    def start(self):
        import sounddevice as sd
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        self._frames.append(indata.copy().flatten())

    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._frames:
            return np.concatenate(self._frames)
        return np.array([], dtype=np.float32)

    def cleanup(self):
        pass


# ── Whisper 로드 ───────────────────────────────────────────────

def load_whisper(model_size: str):
    try:
        from faster_whisper import WhisperModel
        print(f"[Whisper] faster-whisper 로드 중: {model_size} ...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("[Whisper] 준비 완료 (faster-whisper)")
        return model, "faster"
    except ImportError:
        pass

    try:
        import whisper
        print(f"[Whisper] openai-whisper 로드 중: {model_size} ...")
        model = whisper.load_model(model_size)
        print("[Whisper] 준비 완료 (openai-whisper)")
        return model, "openai"
    except ImportError:
        pass

    print("ERROR: Whisper 패키지가 없습니다!")
    print("  pip install faster-whisper  (권장)")
    print("  pip install openai-whisper  (대안)")
    sys.exit(1)


def transcribe(model, backend: str, audio: np.ndarray, lang: Optional[str]) -> str:
    import tempfile, os
    if backend == "faster":
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, audio, RATE)
            kwargs = {"language": lang} if lang else {}
            segments, _ = model.transcribe(tmp, **kwargs)
            return " ".join(s.text for s in segments).strip()
        finally:
            os.unlink(tmp)
    else:
        kwargs = {"language": lang, "fp16": False} if lang else {"fp16": False}
        result = model.transcribe(audio, **kwargs)
        return result.get("text", "").strip()


# ── Linux 음성 명령 파서 ───────────────────────────────────────

def parse_command(text: str) -> Optional[dict]:
    t = text.strip().lower()
    if not t:
        return None

    # ── 앱 열기 ──────────────────────────────────
    m = _match_open(t)
    if m:
        return m

    # ── 앱/창 닫기 ───────────────────────────────
    if _kw(t, ["닫기", "닫아", "창 닫", "close", "close window", "탭 닫"]):
        return {"action": "key", "keys": "ctrl+w", "desc": "창/탭 닫기"}

    # ── 프로그램 종료 ────────────────────────────
    if _kw(t, ["종료", "quit", "앱 종료", "프로그램 종료"]):
        return {"action": "key", "keys": "ctrl+q", "desc": "프로그램 종료"}

    # ── 복사 / 붙여넣기 / 잘라내기 ──────────────
    if _kw(t, ["복사", "copy", "카피"]):
        return {"action": "key", "keys": "ctrl+c", "desc": "복사"}
    if _kw(t, ["붙여넣기", "붙여 넣기", "paste", "페이스트"]):
        return {"action": "key", "keys": "ctrl+v", "desc": "붙여넣기"}
    if _kw(t, ["잘라내기", "잘라 내기", "cut", "컷"]):
        return {"action": "key", "keys": "ctrl+x", "desc": "잘라내기"}

    # ── 실행취소 / 다시실행 ──────────────────────
    if _kw(t, ["취소", "되돌리기", "undo", "언두", "실행취소"]):
        return {"action": "key", "keys": "ctrl+z", "desc": "실행취소"}
    if _kw(t, ["다시 실행", "다시실행", "redo", "리두"]):
        return {"action": "key", "keys": "ctrl+shift+z", "desc": "다시실행"}

    # ── 저장 ─────────────────────────────────────
    if _kw(t, ["저장", "save", "세이브"]):
        return {"action": "key", "keys": "ctrl+s", "desc": "저장"}

    # ── 전체 선택 ────────────────────────────────
    if _kw(t, ["전체 선택", "전체선택", "select all", "모두 선택"]):
        return {"action": "key", "keys": "ctrl+a", "desc": "전체 선택"}

    # ── 찾기 ─────────────────────────────────────
    if _kw(t, ["찾기", "검색", "find", "search"]):
        return {"action": "key", "keys": "ctrl+f", "desc": "찾기"}

    # ── 새 탭 / 새 창 ───────────────────────────
    if _kw(t, ["새 탭", "새탭", "new tab", "뉴탭"]):
        return {"action": "key", "keys": "ctrl+t", "desc": "새 탭"}
    if _kw(t, ["새 창", "새창", "new window", "뉴 윈도우"]):
        return {"action": "key", "keys": "ctrl+n", "desc": "새 창"}

    # ── 탭 전환 ──────────────────────────────────
    if _kw(t, ["다음 탭", "다음탭", "next tab"]):
        return {"action": "key", "keys": "ctrl+Tab", "desc": "다음 탭"}
    if _kw(t, ["이전 탭", "이전탭", "previous tab"]):
        return {"action": "key", "keys": "ctrl+shift+Tab", "desc": "이전 탭"}

    # ── 앱 전환 ──────────────────────────────────
    if _kw(t, ["앱 전환", "앱전환", "switch app", "alt tab", "다른 앱"]):
        return {"action": "key", "keys": "alt+Tab", "desc": "앱 전환"}

    # ── 스크롤 ───────────────────────────────────
    if _kw(t, ["스크롤 다운", "스크롤다운", "scroll down", "아래로", "내려"]):
        return {"action": "scroll", "direction": "down", "desc": "스크롤 다운"}
    if _kw(t, ["스크롤 업", "스크롤업", "scroll up", "위로", "올려"]):
        return {"action": "scroll", "direction": "up", "desc": "스크롤 업"}
    if _kw(t, ["맨 위로", "맨위로", "top"]):
        return {"action": "key", "keys": "Home", "desc": "맨 위로"}
    if _kw(t, ["맨 아래로", "맨아래로", "bottom"]):
        return {"action": "key", "keys": "End", "desc": "맨 아래로"}

    # ── 뒤로 / 앞으로 ───────────────────────────
    if _kw(t, ["뒤로", "뒤로가기", "back", "go back"]):
        return {"action": "key", "keys": "alt+Left", "desc": "뒤로가기"}
    if _kw(t, ["앞으로", "앞으로가기", "forward"]):
        return {"action": "key", "keys": "alt+Right", "desc": "앞으로"}

    # ── 새로고침 ─────────────────────────────────
    if _kw(t, ["새로고침", "refresh", "reload", "리프레시"]):
        return {"action": "key", "keys": "ctrl+r", "desc": "새로고침"}

    # ── 볼륨 ─────────────────────────────────────
    if _kw(t, ["볼륨 업", "볼륨업", "소리 키워", "소리 올려", "volume up", "소리 크게"]):
        return {"action": "volume", "direction": "up", "desc": "볼륨 업"}
    if _kw(t, ["볼륨 다운", "볼륨다운", "소리 줄여", "소리 낮춰", "volume down", "소리 작게"]):
        return {"action": "volume", "direction": "down", "desc": "볼륨 다운"}
    if _kw(t, ["음소거", "뮤트", "mute", "소리 꺼"]):
        return {"action": "volume", "direction": "mute", "desc": "음소거 토글"}

    # ── 밝기 ─────────────────────────────────────
    if _kw(t, ["밝기 올려", "밝기 업", "brightness up", "밝게"]):
        return {"action": "brightness", "direction": "up", "desc": "밝기 올림"}
    if _kw(t, ["밝기 내려", "밝기 다운", "brightness down", "어둡게"]):
        return {"action": "brightness", "direction": "down", "desc": "밝기 내림"}

    # ── 스크린샷 ─────────────────────────────────
    if _kw(t, ["스크린샷", "스크린 샷", "screenshot", "화면 캡처", "캡처"]):
        return {"action": "screenshot", "desc": "스크린샷"}
    if _kw(t, ["영역 캡처", "영역캡처", "부분 캡처"]):
        return {"action": "screenshot_area", "desc": "영역 스크린샷"}

    # ── 화면 잠금 ────────────────────────────────
    if _kw(t, ["화면 잠금", "화면잠금", "잠금", "lock", "lock screen"]):
        return {"action": "lock", "desc": "화면 잠금"}

    # ── 윈도우 ───────────────────────────────────
    if _kw(t, ["최소화", "minimize"]):
        return {"action": "key", "keys": "super+h", "desc": "최소화"}
    if _kw(t, ["전체 화면", "전체화면", "full screen", "풀스크린"]):
        return {"action": "key", "keys": "F11", "desc": "전체 화면"}

    # ── 터미널 ───────────────────────────────────
    if _kw(t, ["터미널", "terminal", "터미널 열어"]):
        return {"action": "open", "target": "gnome-terminal", "desc": "터미널 열기"}

    # ── 파일 매니저 ──────────────────────────────
    if _kw(t, ["파일", "파일 매니저", "files", "파인더", "탐색기"]):
        return {"action": "open", "target": "nautilus", "desc": "파일 매니저"}

    # ── 엔터 / ESC ───────────────────────────────
    if _kw(t, ["엔터", "enter", "확인", "실행"]):
        return {"action": "key", "keys": "Return", "desc": "엔터"}
    if _kw(t, ["이스케이프", "escape", "esc", "나가"]):
        return {"action": "key", "keys": "Escape", "desc": "ESC"}

    return None


# ── 앱 열기 패턴 ────────────────────────────────────────────

_OPEN_PATTERNS = [
    r"(.+?)\s*(?:열어|실행|열기|켜|시작)",
    r"(?:열어|실행|열기|켜)\s*(.+)",
    r"open\s+(.+)",
    r"launch\s+(.+)",
    r"run\s+(.+)",
]

_IS_MAC = sys.platform == "darwin"

_APP_ALIASES_MAC = {
    "chrome":          "Google Chrome",
    "google chrome":   "Google Chrome",
    "크롬":            "Google Chrome",
    "구글 크롬":        "Google Chrome",
    "firefox":         "Firefox",
    "파이어폭스":       "Firefox",
    "safari":          "Safari",
    "사파리":          "Safari",
    "터미널":          "Terminal",
    "terminal":        "Terminal",
    "메모":            "Notes",
    "메모장":          "Notes",
    "계산기":          "Calculator",
    "설정":            "System Settings",
    "파일":            "Finder",
    "finder":          "Finder",
    "음악":            "Music",
    "vscode":          "Visual Studio Code",
    "비주얼 스튜디오":  "Visual Studio Code",
    "브이에스코드":     "Visual Studio Code",
    "슬랙":            "Slack",
    "디스코드":         "Discord",
    "스포티파이":       "Spotify",
    "카카오톡":         "KakaoTalk",
}

_APP_ALIASES_LINUX = {
    "chrome":          "google-chrome",
    "google chrome":   "google-chrome",
    "크롬":            "google-chrome",
    "구글 크롬":        "google-chrome",
    "firefox":         "firefox",
    "파이어폭스":       "firefox",
    "safari":          "firefox",
    "사파리":          "firefox",
    "터미널":          "gnome-terminal",
    "terminal":        "gnome-terminal",
    "메모":            "gedit",
    "메모장":          "gedit",
    "계산기":          "gnome-calculator",
    "설정":            "gnome-control-center",
    "파일":            "nautilus",
    "음악":            "rhythmbox",
    "vscode":          "code",
    "비주얼 스튜디오":  "code",
    "브이에스코드":     "code",
    "슬랙":            "slack",
    "디스코드":         "discord",
    "스포티파이":       "spotify",
}

_APP_ALIASES = _APP_ALIASES_MAC if _IS_MAC else _APP_ALIASES_LINUX


def _match_open(text):
    for pattern in _OPEN_PATTERNS:
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).strip().lower()
            app = _APP_ALIASES.get(raw, raw)
            return {"action": "open", "target": app, "desc": f"{app} 열기"}
    return None


def _kw(text, keywords):
    return any(k in text for k in keywords)


# ── Linux 명령 실행기 ──────────────────────────────────────────

def execute(action: dict) -> bool:
    """액션 실행. 성공하면 True, 실패하면 False."""
    act = action["action"]

    if act == "key":
        return _run(["xdotool", "key", action["keys"]])

    elif act == "open":
        target = action["target"]
        if target.startswith("http"):
            return _run(["xdg-open", target])
        elif sys.platform == "darwin":
            # macOS: open -a "앱이름"
            r = subprocess.run(["open", "-a", target],
                               capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
            else:
                err = r.stderr.decode().strip()
                print(f"    ⚠️  '{target}' 열기 실패: {err}")
                return False
        else:
            # Linux: which로 찾기 → 변형 시도 → xdg-open 폴백
            variants = [
                target,
                f"{target}-stable",
                f"{target}-browser",
            ]
            for v in variants:
                r = subprocess.run(["which", v], capture_output=True)
                if r.returncode == 0:
                    return _run_bg([v])
            print(f"    ⚠️  '{target}' 프로그램을 찾을 수 없습니다")
            return False

    elif act == "scroll":
        btn = "4" if action["direction"] == "up" else "5"
        for _ in range(5):
            _run(["xdotool", "click", btn])
        return True

    elif act == "volume":
        d = action["direction"]
        if d == "up":
            return _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"])
        elif d == "down":
            return _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"])
        elif d == "mute":
            return _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"])

    elif act == "brightness":
        d = action["direction"]
        if d == "up":
            return _run(["xbacklight", "-inc", "10"])
        else:
            return _run(["xbacklight", "-dec", "10"])

    elif act == "screenshot":
        return _run(["gnome-screenshot"])

    elif act == "screenshot_area":
        return _run(["gnome-screenshot", "-a"])

    elif act == "lock":
        return _run(["loginctl", "lock-session"])

    else:
        print(f"    ⚠️  알 수 없는 액션: {act}")
        return False

    return False


def _run(cmd) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=5)
        return r.returncode == 0
    except FileNotFoundError:
        print(f"    ⚠️  명령어 없음: {cmd[0]}")
        return False
    except Exception as e:
        print(f"    ⚠️  실행 실패: {e}")
        return False


def _run_bg(cmd) -> bool:
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        print(f"    ⚠️  프로그램 없음: {cmd[0]}")
        return False
    except Exception as e:
        print(f"    ⚠️  실행 실패: {e}")
        return False


# ── 메인 ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Linux 음성 명령 테스트")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper 모델 크기 (기본: base)")
    parser.add_argument("--lang", default=None,
                        help="언어 고정 (ko=한국어, en=영어, 미지정=자동감지)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="녹음 시간 초 (기본: 3)")
    parser.add_argument("--no-exec", action="store_true",
                        help="명령 인식만, 실행 안 함")
    args = parser.parse_args()

    # Ctrl+C 종료
    signal.signal(signal.SIGINT, lambda *_: (print("\n종료!"), sys.exit(0)))

    print("=" * 55)
    print("  🎙  Linux 음성 명령 테스트")
    print("=" * 55)
    print(f"  모델: {args.model}  |  언어: {args.lang or '자동감지'}")
    print(f"  녹음: {args.duration}초  |  실행: {'OFF' if args.no_exec else 'ON'}")
    print("=" * 55)
    print()

    # 모델 로드
    model, backend = load_whisper(args.model)
    mic = MicRecorder()

    print("사용 가능한 명령어 예시:")
    print("  한국어: 사파리 열어 / 볼륨 업 / 스크린샷 / 복사 / 닫기")
    print("  한국어: 스크롤 다운 / 새 탭 / 저장 / 전체 화면 / 음소거")
    print("  영어:   open chrome / scroll down / copy / screenshot")
    print()
    print("-" * 55)
    print("  Enter = 녹음 시작  |  Ctrl+C = 종료")
    print("-" * 55)
    print()

    count = 0
    while True:
        try:
            input(f"[{count + 1}] Enter를 눌러 녹음 시작 →")
        except EOFError:
            break

        # 녹음
        mic.start()
        print(f"  🔴 녹음 중... ({args.duration}초)")
        time.sleep(args.duration)
        audio = mic.stop()
        print(f"  ⏹  녹음 완료 ({len(audio)} samples, {len(audio)/RATE:.1f}초)")

        # 변환
        print("  🔄 Whisper 변환 중...")
        t0 = time.time()
        text = transcribe(model, backend, audio, args.lang)
        dt = time.time() - t0
        print(f"  📝 인식 결과: \"{text}\"  ({dt:.1f}초)")

        # 명령 매칭
        action = parse_command(text)
        if action:
            print(f"  ✅ 매칭: {action['desc']}")
            print(f"     액션: {action}")
            if not args.no_exec:
                print(f"  🚀 실행 중...")
                ok = execute(action)
                if ok:
                    print(f"  ✅ 완료!")
                else:
                    print(f"  ❌ 실행 실패")
            else:
                print(f"  ⏸  실행 건너뜀 (--no-exec)")
        else:
            print(f"  ❌ 매칭 실패 — 등록된 명령어가 아닙니다")

        count += 1
        print()

    mic.cleanup()
    print("종료!")


if __name__ == "__main__":
    main()
