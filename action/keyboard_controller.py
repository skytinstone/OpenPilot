"""
키보드 컨트롤러 — macOS CGEvent + 클립보드

type_text()  : 클립보드 경유 붙여넣기 (유니코드 전체 지원)
press_keys() : 단축키 조합 실행
open_target(): 앱/URL 열기
"""
import subprocess
import time
from typing import List

# macOS 키코드 (CGEvent)
_KEY_CODES = {
    "cmd":   55,  "command": 55,
    "shift": 56,
    "opt":   58,  "option":  58, "alt": 58,
    "ctrl":  59,  "control": 59,
    "a": 0,  "s": 1,  "d": 2,  "f": 3,  "h": 4,  "g": 5,
    "z": 6,  "x": 7,  "c": 8,  "v": 9,  "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22,
    "5": 23, "=": 24, "9": 25, "7": 26, "-": 27,
    "8": 28, "0": 29,
    "n": 45, "m": 46, "l": 37, "k": 40, "j": 38,
    "i": 34, "o": 31, "u": 32, "p": 35,
    "+": 24, "tab": 48, "space": 49, "return": 36, "esc": 53,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118,
}

_MODIFIER_FLAGS = {
    "cmd":     0x100000,
    "command": 0x100000,
    "shift":   0x020000,
    "opt":     0x080000,
    "option":  0x080000,
    "alt":     0x080000,
    "ctrl":    0x040000,
    "control": 0x040000,
}


# ── 텍스트 입력 ──────────────────────────────────────────────────

def type_text(text: str):
    """
    텍스트를 클립보드에 복사한 뒤 Cmd+V 로 붙여넣기.
    한글 포함 모든 유니코드 문자를 정확하게 입력 가능.
    """
    if not text:
        return
    # 1) 클립보드에 복사
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    time.sleep(0.08)
    # 2) Cmd+V 붙여넣기
    _press_combination(["cmd", "v"])
    time.sleep(0.05)


# ── 단축키 ───────────────────────────────────────────────────────

def press_keys(keys: List[str]):
    """
    keys 리스트로 단축키 실행.
    예: ["cmd", "c"]  →  Cmd+C
        ["cmd", "shift", "t"]  →  Cmd+Shift+T
    """
    _press_combination(keys)


def _press_combination(keys: List[str]):
    from Quartz import (CGEventCreateKeyboardEvent, CGEventSetFlags,
                        CGEventPost, kCGHIDEventTap)

    keys_lower = [k.lower() for k in keys]

    # 수정자 플래그 조합
    flags = 0
    for k in keys_lower:
        flags |= _MODIFIER_FLAGS.get(k, 0)

    # 일반 키 (마지막 비수정자 키)
    normal_keys = [k for k in keys_lower if k not in _MODIFIER_FLAGS]
    if not normal_keys:
        return

    keycode = _KEY_CODES.get(normal_keys[-1], 0)

    for down in [True, False]:
        e = CGEventCreateKeyboardEvent(None, keycode, down)
        CGEventSetFlags(e, flags)
        CGEventPost(kCGHIDEventTap, e)
        time.sleep(0.04)


# ── 앱/URL 열기 ─────────────────────────────────────────────────

def open_target(target: str):
    """
    앱 이름, URL, 파일 경로를 열기.
    예: open_target("Safari")
        open_target("https://google.com")
    """
    try:
        if target.startswith("http://") or target.startswith("https://"):
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["open", "-a", target])
    except Exception as e:
        print(f"[Keyboard] open 실패: {e}")
