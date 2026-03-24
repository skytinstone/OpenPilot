"""
로컬 음성 명령 파서 — API 없이 키워드 매칭으로 PC 제어

Claude API 없이도 기본적인 컴퓨터 제어가 가능합니다.
한국어 / 영어 음성 명령을 키워드 매칭으로 해석하여 액션으로 변환.

지원 명령:
  앱 제어  — 앱 열기/닫기/전환
  텍스트   — 복사/붙여넣기/잘라내기/취소/저장
  시스템   — 볼륨/밝기/스크린샷/잠금
  탐색     — 스크롤/탭 전환/새 탭/뒤로가기
  윈도우   — 최소화/최대화/닫기
"""
import re
from typing import Optional


def parse_voice_command(text: str) -> Optional[dict]:
    """
    음성 텍스트 → 액션 dict 반환.
    매칭되지 않으면 None 반환 (호출부에서 폴백 처리).
    """
    t = text.strip().lower()
    if not t:
        return None

    # ── 앱 열기 ─────────────────────────────────────────────
    m = _match_open(t)
    if m:
        return m

    # ── 앱 닫기 / 창 닫기 ───────────────────────────────────
    if _any_match(t, ["닫기", "닫아", "창 닫", "close", "close window",
                       "탭 닫", "close tab"]):
        return {"action": "shortcut", "keys": ["cmd", "w"]}

    # ── 앱 종료 ─────────────────────────────────────────────
    if _any_match(t, ["종료", "앱 종료", "quit", "quit app", "프로그램 종료"]):
        return {"action": "shortcut", "keys": ["cmd", "q"]}

    # ── 복사 / 붙여넣기 / 잘라내기 ─────────────────────────
    if _any_match(t, ["복사", "copy", "카피"]):
        return {"action": "shortcut", "keys": ["cmd", "c"]}
    if _any_match(t, ["붙여넣기", "붙여 넣기", "paste", "페이스트"]):
        return {"action": "shortcut", "keys": ["cmd", "v"]}
    if _any_match(t, ["잘라내기", "잘라 내기", "cut", "컷"]):
        return {"action": "shortcut", "keys": ["cmd", "x"]}

    # ── 실행취소 / 다시실행 ─────────────────────────────────
    if _any_match(t, ["취소", "되돌리기", "undo", "언두", "실행 취소", "실행취소"]):
        return {"action": "shortcut", "keys": ["cmd", "z"]}
    if _any_match(t, ["다시 실행", "다시실행", "redo", "리두"]):
        return {"action": "shortcut", "keys": ["cmd", "shift", "z"]}

    # ── 저장 ────────────────────────────────────────────────
    if _any_match(t, ["저장", "save", "세이브"]):
        return {"action": "shortcut", "keys": ["cmd", "s"]}

    # ── 전체 선택 ───────────────────────────────────────────
    if _any_match(t, ["전체 선택", "전체선택", "select all", "모두 선택"]):
        return {"action": "shortcut", "keys": ["cmd", "a"]}

    # ── 찾기 ────────────────────────────────────────────────
    if _any_match(t, ["찾기", "검색", "find", "search"]):
        return {"action": "shortcut", "keys": ["cmd", "f"]}

    # ── 새 탭 / 새 창 ──────────────────────────────────────
    if _any_match(t, ["새 탭", "새탭", "new tab", "뉴탭"]):
        return {"action": "shortcut", "keys": ["cmd", "t"]}
    if _any_match(t, ["새 창", "새창", "new window", "뉴 윈도우"]):
        return {"action": "shortcut", "keys": ["cmd", "n"]}

    # ── 탭 전환 ─────────────────────────────────────────────
    if _any_match(t, ["다음 탭", "다음탭", "next tab", "탭 넘기기"]):
        return {"action": "shortcut", "keys": ["ctrl", "tab"]}
    if _any_match(t, ["이전 탭", "이전탭", "previous tab", "탭 뒤로"]):
        return {"action": "shortcut", "keys": ["ctrl", "shift", "tab"]}

    # ── 앱 전환 (Cmd+Tab) ──────────────────────────────────
    if _any_match(t, ["앱 전환", "앱전환", "switch app", "alt tab",
                       "다음 앱", "다른 앱"]):
        return {"action": "shortcut", "keys": ["cmd", "tab"]}

    # ── 스크롤 ──────────────────────────────────────────────
    if _any_match(t, ["스크롤 다운", "스크롤다운", "scroll down",
                       "아래로 스크롤", "아래로", "내려"]):
        return {"action": "scroll", "direction": "down", "amount": 5}
    if _any_match(t, ["스크롤 업", "스크롤업", "scroll up",
                       "위로 스크롤", "위로", "올려"]):
        return {"action": "scroll", "direction": "up", "amount": 5}
    if _any_match(t, ["맨 위로", "맨위로", "top", "scroll to top"]):
        return {"action": "shortcut", "keys": ["cmd", "up"]}
    if _any_match(t, ["맨 아래로", "맨아래로", "bottom", "scroll to bottom"]):
        return {"action": "shortcut", "keys": ["cmd", "down"]}

    # ── 뒤로가기 / 앞으로 ──────────────────────────────────
    if _any_match(t, ["뒤로", "뒤로 가기", "뒤로가기", "back", "go back"]):
        return {"action": "shortcut", "keys": ["cmd", "left"]}
    if _any_match(t, ["앞으로", "앞으로 가기", "앞으로가기", "forward", "go forward"]):
        return {"action": "shortcut", "keys": ["cmd", "right"]}

    # ── 새로고침 ────────────────────────────────────────────
    if _any_match(t, ["새로고침", "새로 고침", "refresh", "reload", "리프레시"]):
        return {"action": "shortcut", "keys": ["cmd", "r"]}

    # ── 볼륨 ────────────────────────────────────────────────
    if _any_match(t, ["볼륨 업", "볼륨업", "소리 키워", "소리 올려",
                       "volume up", "소리 크게"]):
        return {"action": "system", "command": "volume_up"}
    if _any_match(t, ["볼륨 다운", "볼륨다운", "소리 줄여", "소리 낮춰",
                       "volume down", "소리 작게"]):
        return {"action": "system", "command": "volume_down"}
    if _any_match(t, ["음소거", "뮤트", "mute", "소리 꺼"]):
        return {"action": "system", "command": "mute"}

    # ── 밝기 ────────────────────────────────────────────────
    if _any_match(t, ["밝기 올려", "밝기 업", "밝기업", "brightness up",
                       "밝게", "화면 밝게"]):
        return {"action": "system", "command": "brightness_up"}
    if _any_match(t, ["밝기 내려", "밝기 다운", "밝기다운", "brightness down",
                       "어둡게", "화면 어둡게"]):
        return {"action": "system", "command": "brightness_down"}

    # ── 스크린샷 ────────────────────────────────────────────
    if _any_match(t, ["스크린샷", "스크린 샷", "screenshot", "화면 캡처",
                       "화면캡처", "캡처"]):
        return {"action": "shortcut", "keys": ["cmd", "shift", "3"]}
    if _any_match(t, ["영역 캡처", "영역캡처", "부분 캡처", "부분캡처",
                       "영역 스크린샷"]):
        return {"action": "shortcut", "keys": ["cmd", "shift", "4"]}

    # ── 화면 잠금 ───────────────────────────────────────────
    if _any_match(t, ["화면 잠금", "화면잠금", "잠금", "lock", "lock screen"]):
        return {"action": "shortcut", "keys": ["cmd", "ctrl", "q"]}

    # ── 윈도우 제어 ─────────────────────────────────────────
    if _any_match(t, ["최소화", "minimize", "미니마이즈"]):
        return {"action": "shortcut", "keys": ["cmd", "m"]}
    if _any_match(t, ["전체 화면", "전체화면", "full screen", "풀스크린"]):
        return {"action": "shortcut", "keys": ["cmd", "ctrl", "f"]}
    if _any_match(t, ["화면 숨기기", "숨기기", "hide", "숨겨"]):
        return {"action": "shortcut", "keys": ["cmd", "h"]}

    # ── Spotlight / 런처 ────────────────────────────────────
    if _any_match(t, ["스팟라이트", "spotlight", "런처", "검색창"]):
        return {"action": "shortcut", "keys": ["cmd", "space"]}

    # ── Mission Control / Exposé ────────────────────────────
    if _any_match(t, ["미션 컨트롤", "미션컨트롤", "mission control",
                       "모든 창", "모든창"]):
        return {"action": "shortcut", "keys": ["ctrl", "up"]}

    # ── 엔터 / 스페이스 / ESC ───────────────────────────────
    if _any_match(t, ["엔터", "enter", "확인", "실행"]):
        return {"action": "shortcut", "keys": ["return"]}
    if _any_match(t, ["이스케이프", "escape", "esc", "취소해", "나가"]):
        return {"action": "shortcut", "keys": ["esc"]}
    if _any_match(t, ["스페이스", "space", "공백"]):
        return {"action": "shortcut", "keys": ["space"]}

    # ── 삭제 (백스페이스) ───────────────────────────────────
    if _any_match(t, ["삭제", "지워", "delete", "backspace"]):
        return {"action": "system", "command": "delete"}

    # ── 매칭 실패 ───────────────────────────────────────────
    return None


# ── 앱 열기 패턴 매칭 ──────────────────────────────────────────

_OPEN_PATTERNS = [
    # 한국어
    r"(.+?)\s*(?:열어|실행|열기|켜|시작)",
    r"(?:열어|실행|열기|켜)\s*(.+)",
    # 영어
    r"open\s+(.+)",
    r"launch\s+(.+)",
    r"start\s+(.+)",
    r"run\s+(.+)",
]

# 앱 이름 정규화 (한국어 → 실제 앱 이름)
_APP_ALIASES = {
    "safari": "Safari",
    "사파리": "Safari",
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "크롬": "Google Chrome",
    "구글 크롬": "Google Chrome",
    "firefox": "Firefox",
    "파이어폭스": "Firefox",
    "finder": "Finder",
    "파인더": "Finder",
    "terminal": "Terminal",
    "터미널": "Terminal",
    "메모": "Notes",
    "메모장": "Notes",
    "캘린더": "Calendar",
    "달력": "Calendar",
    "계산기": "Calculator",
    "설정": "System Preferences",
    "시스템 설정": "System Settings",
    "환경설정": "System Settings",
    "메일": "Mail",
    "음악": "Music",
    "뮤직": "Music",
    "메시지": "Messages",
    "카카오톡": "KakaoTalk",
    "슬랙": "Slack",
    "디스코드": "Discord",
    "비주얼 스튜디오": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "브이에스코드": "Visual Studio Code",
    "엑셀": "Microsoft Excel",
    "워드": "Microsoft Word",
    "파워포인트": "Microsoft PowerPoint",
    "피피티": "Microsoft PowerPoint",
    "줌": "zoom.us",
    "스포티파이": "Spotify",
    "앱스토어": "App Store",
    "활성 상태 보기": "Activity Monitor",
    "액티비티 모니터": "Activity Monitor",
}


def _match_open(text: str) -> Optional[dict]:
    for pattern in _OPEN_PATTERNS:
        m = re.search(pattern, text)
        if m:
            app_name = m.group(1).strip()
            # 별칭 변환
            app_name = _APP_ALIASES.get(app_name, app_name)
            # 첫 글자 대문자로
            if app_name and app_name[0].isascii():
                app_name = app_name.title()
            return {"action": "open", "target": app_name}
    return None


# ── 유틸리티 ───────────────────────────────────────────────────

def _any_match(text: str, keywords: list) -> bool:
    """키워드 중 하나라도 텍스트에 포함되면 True"""
    for kw in keywords:
        if kw in text:
            return True
    return False
