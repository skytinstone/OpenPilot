"""
AI 명령 처리 — Claude API (anthropic)

음성으로 받은 텍스트를 Claude 가 해석해 실행 가능한 액션으로 변환.

지원 액션:
  type      — 현재 포커스된 앱에 텍스트 입력
  shortcut  — 키보드 단축키 실행
  scroll    — 스크롤 (up/down)
  click     — 마우스 클릭 (left/right)
  open      — 앱/URL 열기
  say       — AI 응답만 반환 (실행 없음)
"""
import json
import os
from typing import Optional

_SYSTEM_PROMPT = """
You are an AI computer control assistant integrated into OpenPilot,
a hands-free computer control system.

The user will give you a voice command (transcribed speech).
Interpret the command and return a single JSON object with the action to execute.

Available actions:
  {"action": "type",     "text": "text to type into the focused app"}
  {"action": "shortcut", "keys": ["cmd", "c"]}
  {"action": "scroll",   "direction": "up" | "down", "amount": 3}
  {"action": "click",    "button": "left" | "right"}
  {"action": "open",     "target": "Safari" | "https://..." | "/path/to/app"}
  {"action": "say",      "text": "response text when no action applies"}

Rules:
- Return ONLY valid JSON. No markdown, no explanation.
- If the command is ambiguous, choose the most likely intent.
- If the command is to dictate/write something, use "type".
- Common shortcuts: copy=["cmd","c"], paste=["cmd","v"], undo=["cmd","z"],
  save=["cmd","s"], new tab=["cmd","t"], close=["cmd","w"]
- For "zoom in/out" use shortcut ["cmd", "+"] or ["cmd", "-"].
"""


class CommandProcessor:
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-6"):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.\n"
                "  export ANTHROPIC_API_KEY=your_key_here"
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=key)
        self._model  = model

    def process(self, transcription: str) -> dict:
        """
        음성 텍스트 → Claude 해석 → 액션 dict 반환.
        실패 시 {"action": "type", "text": transcription} 폴백.
        """
        if not transcription.strip():
            return {"action": "say", "text": ""}

        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": transcription}],
            )
            raw = msg.content[0].text.strip()
            # JSON 블록 정리 (```json ... ``` 형태 대비)
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as e:
            print(f"[AI] 처리 실패: {e} — 텍스트 입력으로 폴백")
            return {"action": "type", "text": transcription}
