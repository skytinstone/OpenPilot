"""
Gemini 3 AI 에이전트 — OpenPilot 내장 PC 제어 AI

API 키만 설정하면 OpenClaw 없이도 자연어로 PC를 제어합니다.

파이프라인:
  음성 텍스트 → Gemini 3 API → JSON 액션 → 실행

지원 액션:
  open      — 앱/URL 열기
  shortcut  — 키보드 단축키
  type      — 텍스트 입력
  scroll    — 스크롤
  system    — 볼륨/밝기 등
  shell     — 셸 명령 실행
  say       — 응답만 (실행 없음)
"""
import json
import os
import re
import subprocess
from typing import Optional

# API 키 로드 경로
_KEYS_PATH = os.path.join(os.path.dirname(__file__), "../../config/api_keys.json")

# macOS 용 시스템 프롬프트
_SYSTEM_PROMPT = """You are OpenPilot AI, a voice-controlled PC assistant running on macOS.

The user will give voice commands in Korean or English. You MUST return a single JSON action.

## Available Actions

1. **open** — Open an app or URL
   ```json
   {"action": "open", "target": "Google Chrome"}
   {"action": "open", "target": "https://google.com"}
   ```

2. **shortcut** — Keyboard shortcut (macOS keys: cmd, shift, opt, ctrl, tab, return, esc, space, up, down, left, right, delete, F1-F12)
   ```json
   {"action": "shortcut", "keys": ["cmd", "c"]}
   {"action": "shortcut", "keys": ["cmd", "shift", "4"]}
   ```

3. **type** — Type text into the focused app
   ```json
   {"action": "type", "text": "Hello World"}
   ```

4. **scroll** — Scroll up or down
   ```json
   {"action": "scroll", "direction": "down", "amount": 5}
   ```

5. **system** — System controls
   ```json
   {"action": "system", "command": "volume_up"}
   {"action": "system", "command": "volume_down"}
   {"action": "system", "command": "mute"}
   {"action": "system", "command": "brightness_up"}
   {"action": "system", "command": "brightness_down"}
   ```

6. **shell** — Run a shell command (for anything not covered above)
   ```json
   {"action": "shell", "command": "say '안녕하세요'"}
   {"action": "shell", "command": "open -a 'Activity Monitor'"}
   ```

7. **say** — Just reply to the user (no action needed)
   ```json
   {"action": "say", "text": "현재 시간은 오후 3시입니다."}
   ```

## Rules
- ALWAYS return exactly one JSON object. No markdown, no explanation, just JSON.
- For Korean app names, use the macOS app name:
  - 크롬/Chrome → "Google Chrome"
  - 사파리 → "Safari"
  - 터미널 → "Terminal"
  - 파인더 → "Finder"
  - 메모 → "Notes"
  - 캘린더 → "Calendar"
  - 카카오톡 → "KakaoTalk"
  - 슬랙 → "Slack"
- For web searches: {"action": "shell", "command": "open 'https://www.google.com/search?q=검색어'"}
- When asked for time/date: use shell with "date" command or just say it
- Be smart about interpreting Korean commands
"""


class GeminiAgent:
    """Gemini 3 API를 직접 호출하는 PC 제어 에이전트"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gemini-2.5-flash"):
        self._api_key = api_key or self._load_key()
        self._model_name = model
        self._model = None
        self._chat = None

        if not self._api_key:
            raise ValueError(
                "Gemini API 키가 없습니다.\n"
                "  python main.py --setup 으로 설정하거나\n"
                "  export GEMINI_API_KEY=your_key 를 실행하세요."
            )
        self._init_model()

    def _load_key(self) -> str:
        """환경변수 → config/api_keys.json 순으로 키 탐색"""
        # 1. 환경변수
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            return key

        # 2. config 파일
        try:
            path = os.path.normpath(_KEYS_PATH)
            with open(path) as f:
                data = json.load(f)
            return data.get("gemini", "")
        except Exception:
            return ""

    def _init_model(self):
        """Gemini 모델 초기화"""
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=_SYSTEM_PROMPT,
        )
        # 대화 히스토리 유지 (문맥 이해)
        self._chat = self._model.start_chat(history=[])
        print(f"[GeminiAgent] 모델 준비 완료: {self._model_name}")

    def process(self, text: str) -> dict:
        """
        자연어 명령 → JSON 액션 dict 반환.
        실패 시 {"action": "say", "text": "..."} 반환.
        """
        if not text.strip():
            return {"action": "say", "text": ""}

        try:
            response = self._chat.send_message(text)
            raw = response.text.strip()
            return self._parse_json(raw)
        except Exception as e:
            print(f"[GeminiAgent] API 오류: {e}")
            return {"action": "say", "text": f"AI 오류: {e}"}

    def _parse_json(self, raw: str) -> dict:
        """Gemini 응답에서 JSON 추출"""
        # 1. 직접 JSON 파싱 시도
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2. 코드 블록에서 JSON 추출
        m = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 3. 중괄호 영역 추출
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # 4. 파싱 실패 → 응답 텍스트 그대로 반환
        return {"action": "say", "text": raw}


def is_available() -> bool:
    """Gemini API 사용 가능 여부 확인"""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return True
    try:
        path = os.path.normpath(_KEYS_PATH)
        with open(path) as f:
            data = json.load(f)
        return bool(data.get("gemini", ""))
    except Exception:
        return False


def save_api_key(provider: str, key: str):
    """API 키를 config/api_keys.json에 저장"""
    path = os.path.normpath(_KEYS_PATH)
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        data = {}

    data[provider] = key
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Config] {provider} API 키 저장 완료: {path}")
