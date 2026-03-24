"""
OpenClaw 브릿지 — 음성 명령을 OpenClaw AI 에이전트에 전달

OpenPilot 음성 → Whisper 텍스트 → OpenClaw agent → Gemini 3 실행

지원 기능:
  - 앱 열기/닫기       "크롬 열어줘", "터미널 열어"
  - 웹 검색            "구글에서 OOO 검색해줘"
  - 파일 관리          "다운로드 폴더 열어", "파일 찾아줘"
  - 시스템 제어        "볼륨 올려", "밝기 낮춰"
  - 질문/답변          "오늘 날씨 알려줘", "환율 알려줘"
  - 메모/리마인더      "메모 작성해줘", "알람 설정해줘"
  - 코드/개발          "이 코드 설명해줘", "PR 만들어줘"
  - 자유 대화          자연어로 무엇이든 요청
"""
import subprocess
import os
import threading
from typing import Optional, Callable


def is_available() -> bool:
    """OpenClaw CLI가 설치되어 있는지 확인"""
    try:
        r = subprocess.run(["which", "openclaw"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def has_api_key() -> bool:
    """Gemini API 키가 환경변수에 설정되어 있는지 확인"""
    return bool(os.environ.get("GEMINI_API_KEY", ""))


def send_command(message: str, timeout: int = 30) -> str:
    """
    OpenClaw 에이전트에 메시지 전송 → 응답 반환.
    동기 호출 (블로킹).
    """
    if not message.strip():
        return ""

    env = os.environ.copy()

    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "main", "--message", message],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = result.stdout.strip()
        # Gateway 폴백 메시지 제거
        lines = output.split("\n")
        cleaned = []
        for line in lines:
            if any(skip in line for skip in [
                "Gateway agent failed",
                "Gateway target:",
                "Source:",
                "Config:",
                "Bind:",
            ]):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()
    except subprocess.TimeoutExpired:
        return "[OpenClaw] 응답 시간 초과 (30초)"
    except FileNotFoundError:
        return "[OpenClaw] CLI를 찾을 수 없습니다. npm i -g openclaw"
    except Exception as e:
        return f"[OpenClaw] 오류: {e}"


def send_command_async(message: str,
                       on_done: Callable[[str], None],
                       timeout: int = 30):
    """
    OpenClaw 에이전트에 비동기 전송.
    완료 시 on_done(response_text) 콜백 호출.
    """
    def _worker():
        response = send_command(message, timeout=timeout)
        on_done(response)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
